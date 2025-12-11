"""
Jina v2 Embedding Model Fine-tuning Trainer
Implements contrastive learning for embedding fine-tuning
"""
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config

# Ensure all HuggingFace cache and outputs go to data disk
os.environ["HF_HOME"] = str(config.MODEL_DIR.parent / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(config.MODEL_DIR)
os.environ["HF_DATASETS_CACHE"] = str(config.CACHE_DIR / "datasets")
os.environ["TORCH_HOME"] = str(config.CACHE_DIR / "torch")

try:
    from transformers import (
        AutoTokenizer,
        AutoModel,
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup
    )
    from torch.optim import AdamW
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with multi-task support"""
    
    def __init__(self, training_examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = training_examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        positive = example["positive"]
        negatives = example["negatives"]
        task_weight = example.get("task_weight", 1.0)  # Default weight is 1.0
        task_type = positive.get("task_type", "qa")  # Default task type is "qa"
        
        # Tokenize anchor and positive
        anchor_text = positive["anchor"]
        positive_text = positive["positive"]
        
        anchor_encoded = self.tokenizer(
            anchor_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        positive_encoded = self.tokenizer(
            positive_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize negatives
        negative_texts = [neg["negative"] for neg in negatives]
        negative_encoded = self.tokenizer(
            negative_texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'anchor_input_ids': anchor_encoded['input_ids'].squeeze(),
            'anchor_attention_mask': anchor_encoded['attention_mask'].squeeze(),
            'positive_input_ids': positive_encoded['input_ids'].squeeze(),
            'positive_attention_mask': positive_encoded['attention_mask'].squeeze(),
            'negative_input_ids': negative_encoded['input_ids'],
            'negative_attention_mask': negative_encoded['attention_mask'],
            'task_weight': torch.tensor(task_weight, dtype=torch.float32),
            'task_type': task_type
        }


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss for contrastive learning with in-batch negatives support
    Based on SimCSE, DPR, and CLIP best practices
    """
    
    def __init__(self, temperature: float = 0.05, use_in_batch_negatives: bool = True):
        super().__init__()
        self.temperature = temperature
        self.use_in_batch_negatives = use_in_batch_negatives
    
    def forward(self, anchor_emb, positive_emb, negative_embs=None):
        """
        Compute InfoNCE loss with optional in-batch negatives
        
        Args:
            anchor_emb: Anchor embeddings [batch_size, hidden_size]
            positive_emb: Positive embeddings [batch_size, hidden_size]
            negative_embs: Optional negative embeddings [batch_size, num_negatives, hidden_size]
                          If None and use_in_batch_negatives=True, uses batch negatives
        """
        # Normalize embeddings
        anchor_emb = nn.functional.normalize(anchor_emb, p=2, dim=1)
        positive_emb = nn.functional.normalize(positive_emb, p=2, dim=1)
        
        batch_size = anchor_emb.size(0)
        
        # Positive similarity: anchor with its positive
        pos_sim = torch.sum(anchor_emb * positive_emb, dim=1) / self.temperature  # [batch_size]
        
        # Collect all negative similarities
        all_neg_sims = []
        
        # 1. Explicit negatives (if provided)
        if negative_embs is not None:
            negative_embs = nn.functional.normalize(negative_embs, p=2, dim=2)
            anchor_expanded = anchor_emb.unsqueeze(1).expand(-1, negative_embs.size(1), -1)
            explicit_neg_sims = torch.sum(anchor_expanded * negative_embs, dim=2) / self.temperature
            all_neg_sims.append(explicit_neg_sims)
        
        # 2. In-batch negatives (use other positives in batch as negatives)
        if self.use_in_batch_negatives:
            # Compute similarity matrix: [batch_size, batch_size]
            # Each row i: similarity of anchor[i] with all positives in batch
            batch_sim_matrix = torch.matmul(anchor_emb, positive_emb.t()) / self.temperature  # [batch_size, batch_size]
            
            # Remove diagonal (self-similarity) to get negatives
            # Create mask to exclude diagonal
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=anchor_emb.device)
            batch_neg_sims = batch_sim_matrix[mask].view(batch_size, batch_size - 1)  # [batch_size, batch_size-1]
            all_neg_sims.append(batch_neg_sims)
        
        # Concatenate all negatives
        if all_neg_sims:
            all_negatives = torch.cat(all_neg_sims, dim=1)  # [batch_size, total_negatives]
        else:
            # Fallback: if no negatives, use zeros
            all_negatives = torch.zeros(batch_size, 1, device=anchor_emb.device)
        
        # Concatenate positive and all negatives
        logits = torch.cat([pos_sim.unsqueeze(1), all_negatives], dim=1)  # [batch_size, 1 + total_negatives]
        
        # Labels: 0 is positive (first position)
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor_emb.device)
        
        # Cross entropy loss
        loss = nn.functional.cross_entropy(logits, labels)
        
        return loss


class JinaTrainer:
    """
    Jina v2 Embedding Model Fine-tuning Trainer
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        learning_rate: float = None,
        batch_size: int = None,
        max_length: int = None,
        gradient_accumulation_steps: int = None,
        temperature: float = None,
        use_in_batch_negatives: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            model_name: Base model name
            device: Device to use
            learning_rate: Learning rate
            batch_size: Batch size
            max_length: Max sequence length
            gradient_accumulation_steps: Number of gradient accumulation steps
            temperature: Temperature parameter for InfoNCE loss (default: 0.05)
            use_in_batch_negatives: Whether to use in-batch negatives (default: True)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required")
        
        self.model_name = model_name or config.JINA_BASE_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.learning_rate = learning_rate or config.TRAINING_LEARNING_RATE
        self.batch_size = batch_size or config.TRAINING_BATCH_SIZE
        self.max_length = max_length or config.TRAINING_MAX_LENGTH
        self.gradient_accumulation_steps = gradient_accumulation_steps or config.TRAINING_GRADIENT_ACCUMULATION_STEPS
        
        # Load tokenizer and model
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=str(config.MODEL_DIR)
        )
        
        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=str(config.MODEL_DIR)
        )
        
        self.model.to(self.device)
        
        # Loss function with configurable temperature and in-batch negatives
        temperature = temperature if temperature is not None else getattr(config, 'TRAINING_TEMPERATURE', 0.05)
        self.criterion = InfoNCELoss(
            temperature=temperature,
            use_in_batch_negatives=use_in_batch_negatives
        )
        
        print(f"✅ Trainer initialized on {self.device}")
        print(f"   Temperature: {temperature}")
        print(f"   In-batch negatives: {use_in_batch_negatives}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def evaluate(self, eval_data_path: Path) -> Dict:
        """
        Evaluate model on validation set
        
        Args:
            eval_data_path: Path to evaluation data JSON file
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        # Load evaluation data
        with open(eval_data_path, 'r', encoding='utf-8') as f:
            eval_examples = json.load(f)
        
        # Create dataset and dataloader
        eval_dataset = ContrastiveDataset(eval_examples, self.tokenizer, self.max_length)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move to device
                anchor_input_ids = batch['anchor_input_ids'].to(self.device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                negative_input_ids = batch['negative_input_ids'].to(self.device)
                negative_attention_mask = batch['negative_attention_mask'].to(self.device)
                
                # Forward pass - anchor
                anchor_output = self.model(
                    input_ids=anchor_input_ids,
                    attention_mask=anchor_attention_mask
                )
                anchor_emb = self._mean_pooling(anchor_output, anchor_attention_mask)
                
                # Forward pass - positive
                positive_output = self.model(
                    input_ids=positive_input_ids,
                    attention_mask=positive_attention_mask
                )
                positive_emb = self._mean_pooling(positive_output, positive_attention_mask)
                
                # Forward pass - negatives
                batch_size, num_negatives, seq_len = negative_input_ids.shape
                negative_input_ids_flat = negative_input_ids.view(-1, seq_len)
                negative_attention_mask_flat = negative_attention_mask.view(-1, seq_len)
                
                negative_output = self.model(
                    input_ids=negative_input_ids_flat,
                    attention_mask=negative_attention_mask_flat
                )
                negative_embs = self._mean_pooling(negative_output, negative_attention_mask_flat)
                negative_embs = negative_embs.view(batch_size, num_negatives, -1)
                
                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb, negative_embs)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        self.model.train()
        
        return {
            "eval_loss": avg_loss,
            "num_samples": len(eval_examples)
        }
    
    def train(
        self,
        training_data_path: Path,
        output_dir: Path,
        epochs: int = None,
        eval_data_path: Optional[Path] = None,
        save_steps: int = None,
        eval_steps: int = None
    ):
        """
        Train the model
        
        Args:
            training_data_path: Path to training data JSON file
            output_dir: Output directory for checkpoints
            epochs: Number of training epochs
            eval_data_path: Path to evaluation data (optional)
            save_steps: Steps between checkpoints
            eval_steps: Steps between evaluations
        """
        epochs = epochs or config.TRAINING_EPOCHS
        save_steps = save_steps or config.TRAINING_SAVE_STEPS
        eval_steps = eval_steps or config.TRAINING_EVAL_STEPS
        
        # Ensure output_dir is on data disk (/root/autodl-tmp/)
        output_dir = Path(output_dir)
        if not str(output_dir).startswith("/root/autodl-tmp/"):
            # If relative path or not on data disk, use TRAINING_OUTPUT_DIR
            print(f"⚠️ Warning: Output directory {output_dir} is not on data disk.")
            print(f"   Redirecting to data disk: {config.TRAINING_OUTPUT_DIR / output_dir.name}")
            output_dir = config.TRAINING_OUTPUT_DIR / output_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 All training outputs will be saved to: {output_dir}")
        
        # Load training data
        print(f"Loading training data from: {training_data_path}")
        with open(training_data_path, 'r', encoding='utf-8') as f:
            training_examples = json.load(f)
        
        # Create dataset and dataloader
        dataset = ContrastiveDataset(training_examples, self.tokenizer, self.max_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        # Adjust total steps for gradient accumulation
        # Each optimizer step processes gradient_accumulation_steps batches
        steps_per_epoch = (len(dataloader) + self.gradient_accumulation_steps - 1) // self.gradient_accumulation_steps
        total_steps = steps_per_epoch * epochs
        # Use cosine annealing with warmup for smoother learning rate decay
        # Calculate warmup steps: use ratio-based or fixed steps (whichever is larger)
        warmup_ratio = getattr(config, 'TRAINING_WARMUP_RATIO', 0.1)
        warmup_steps_from_ratio = int(total_steps * warmup_ratio)
        warmup_steps = max(config.TRAINING_WARMUP_STEPS, warmup_steps_from_ratio)
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5  # Half cosine cycle for smoother decay
        )
        
        print(f"📊 Training Configuration:")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"   Effective batch size: {self.batch_size * self.gradient_accumulation_steps}")
        print(f"   Epochs: {epochs}")
        print(f"   Total steps: {total_steps}")
        if eval_data_path:
            print(f"   Validation set: {eval_data_path}")
            print(f"   Early stopping patience: {config.TRAINING_EARLY_STOPPING_PATIENCE}")
            print(f"   Save best model: {config.TRAINING_SAVE_BEST_MODEL}")
        
        # Training loop
        self.model.train()
        global_step = 0
        
        # Early stopping and best model tracking
        best_eval_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        training_history = []
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}")
            accumulated_loss = 0
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                anchor_input_ids = batch['anchor_input_ids'].to(self.device)
                anchor_attention_mask = batch['anchor_attention_mask'].to(self.device)
                positive_input_ids = batch['positive_input_ids'].to(self.device)
                positive_attention_mask = batch['positive_attention_mask'].to(self.device)
                negative_input_ids = batch['negative_input_ids'].to(self.device)
                negative_attention_mask = batch['negative_attention_mask'].to(self.device)
                
                # Get task weights for multi-task learning (if available)
                task_weights = batch.get('task_weight', torch.ones(anchor_input_ids.size(0))).to(self.device)
                
                # Forward pass - anchor
                anchor_output = self.model(
                    input_ids=anchor_input_ids,
                    attention_mask=anchor_attention_mask
                )
                anchor_emb = self._mean_pooling(anchor_output, anchor_attention_mask)
                
                # Forward pass - positive
                positive_output = self.model(
                    input_ids=positive_input_ids,
                    attention_mask=positive_attention_mask
                )
                positive_emb = self._mean_pooling(positive_output, positive_attention_mask)
                
                # Forward pass - negatives
                batch_size, num_negatives, seq_len = negative_input_ids.shape
                negative_input_ids_flat = negative_input_ids.view(-1, seq_len)
                negative_attention_mask_flat = negative_attention_mask.view(-1, seq_len)
                
                negative_output = self.model(
                    input_ids=negative_input_ids_flat,
                    attention_mask=negative_attention_mask_flat
                )
                negative_embs = self._mean_pooling(negative_output, negative_attention_mask_flat)
                negative_embs = negative_embs.view(batch_size, num_negatives, -1)
                
                # Compute loss
                loss = self.criterion(anchor_emb, positive_emb, negative_embs)
                
                # Apply task weights for multi-task learning
                # If task_weights are different, we need to compute per-sample loss
                if task_weights.size(0) > 1 and not torch.allclose(task_weights, task_weights[0]):
                    # Compute per-sample losses (requires more computation)
                    # For efficiency, we'll use batch-level weighting
                    loss = loss * task_weights.mean()
                
                loss_value = loss.item()
                
                # Normalize loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass (accumulate gradients)
                loss.backward()
                
                accumulated_loss += loss_value
                
                # Update weights only after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_loss += accumulated_loss
                    global_step += 1
                    
                    # Display average loss over accumulation steps
                    avg_loss_display = accumulated_loss / self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        'loss': f'{avg_loss_display:.4f}',
                        'step': global_step
                    })
                    
                    accumulated_loss = 0
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(checkpoint_dir)
                        self.tokenizer.save_pretrained(checkpoint_dir)
                        print(f"\n💾 Saved checkpoint to {checkpoint_dir}")
            
            # Handle remaining gradients if batch doesn't divide evenly
            if accumulated_loss > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += accumulated_loss
                global_step += 1
            
            avg_loss = epoch_loss / len(dataloader)
            print(f"\n📊 Training loss for epoch {epoch + 1}: {avg_loss:.4f}")
            
            # Evaluate on validation set
            eval_loss = None
            if eval_data_path and eval_data_path.exists():
                eval_results = self.evaluate(eval_data_path)
                eval_loss = eval_results["eval_loss"]
                print(f"📊 Validation loss for epoch {epoch + 1}: {eval_loss:.4f}")
                
                # Track training history
                training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_loss": eval_loss
                })
                
                # Check for improvement
                improvement = best_eval_loss - eval_loss
                if improvement > config.TRAINING_EARLY_STOPPING_MIN_DELTA:
                    best_eval_loss = eval_loss
                    best_epoch = epoch + 1
                    patience_counter = 0
                    
                    # Save best model
                    if config.TRAINING_SAVE_BEST_MODEL:
                        best_model_dir = output_dir / "best_model"
                        best_model_dir.mkdir(parents=True, exist_ok=True)
                        self.model.save_pretrained(best_model_dir)
                        self.tokenizer.save_pretrained(best_model_dir)
                        print(f"✅ Best model saved! (Epoch {best_epoch}, Eval Loss: {best_eval_loss:.4f})")
                else:
                    patience_counter += 1
                    print(f"⚠️ No improvement for {patience_counter} epoch(s)")
                
                # Early stopping check
                if patience_counter >= config.TRAINING_EARLY_STOPPING_PATIENCE:
                    print(f"\n🛑 Early stopping triggered! No improvement for {config.TRAINING_EARLY_STOPPING_PATIENCE} epochs.")
                    print(f"   Best epoch: {best_epoch}, Best eval loss: {best_eval_loss:.4f}")
                    break
            else:
                training_history.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "eval_loss": None
                })
        
        # Save training history
        history_path = output_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(training_history, f, indent=2, ensure_ascii=False)
        print(f"\n📝 Training history saved to: {history_path}")
        
        # Save final model
        print(f"\n💾 Saving final model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Summary
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Total epochs trained: {len(training_history)}")
        if eval_data_path and eval_data_path.exists():
            print(f"Best epoch: {best_epoch}")
            print(f"Best validation loss: {best_eval_loss:.4f}")
            print(f"Best model saved to: {output_dir / 'best_model'}")
        print(f"Final model saved to: {output_dir}")
        print("✅ Training completed!")

