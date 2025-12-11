"""
Training Data Preparation Module
Prepares question-answer pairs and question-document pairs for Jina model training
Supports multi-task learning (QA similarity + QD retrieval similarity)
"""
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.data.data_loader import ScienceQADataLoader
from src.rag.vector_store import ScienceQAVectorStore


class TrainingDataPreparation:
    """
    Prepare training data for Jina v2 embedding fine-tuning
    Generates positive/negative pairs for contrastive learning
    """
    
    def __init__(self, data_loader: Optional[ScienceQADataLoader] = None, use_hard_negatives: bool = True):
        """
        Initialize data preparation
        
        Args:
            data_loader: ScienceQA data loader instance
            use_hard_negatives: Whether to use hard negative mining (requires embedding model)
        """
        self.data_loader = data_loader or ScienceQADataLoader()
        self.problems = None
        self.captions = None
        self.splits = None
        self.use_hard_negatives = use_hard_negatives
        self.vector_store = None
        self.embedding_model = None
        self.problem_documents = {}  # Cache for problem documents
    
    def load_data(self):
        """Load ScienceQA dataset"""
        print("Loading ScienceQA dataset...")
        self.problems = self.data_loader.load_problems()
        self.captions = self.data_loader.load_captions()
        self.splits = self.data_loader.load_pid_splits()
        print(f"Loaded {len(self.problems)} problems")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for hard negative mining"""
        if not self.use_hard_negatives:
            return
        
        try:
            # Use a lightweight embedding model for hard negative mining
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=str(config.MODEL_DIR)
            )
            print("✅ Embedding model initialized for hard negative mining")
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize embedding model for hard negatives: {e}")
            print("   Falling back to random negative sampling")
            self.use_hard_negatives = False
    
    def generate_question_answer_pairs(
        self,
        split: str = "train",
        include_image_context: bool = True
    ) -> List[Dict]:
        """
        Generate question-answer pairs for training
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            include_image_context: Whether to include image captions
            
        Returns:
            List of question-answer pairs
        """
        if self.problems is None:
            self.load_data()
        
        split_problems = self.data_loader.get_split_problems(split)
        pairs = []
        
        for pid, problem in split_problems.items():
            question = problem.get("question", "")
            answer = problem.get("answer", "")
            choices = problem.get("choices", [])
            explanation = problem.get("explanation") or problem.get("solution", "")
            
            if not question or not answer:
                continue
            
            # Build question text
            question_text = question
            if choices:
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                question_text += f"\n{choices_text}"
            
            # Build answer text
            answer_text = ""
            if choices and isinstance(answer, (int, str)):
                try:
                    answer_idx = int(answer) if isinstance(answer, str) else answer
                    if 0 <= answer_idx < len(choices):
                        answer_text = choices[answer_idx]
                except:
                    answer_text = str(answer)
            
            if explanation:
                answer_text += f"\n{explanation}"
            
            # Add image context if available
            image_context = ""
            if include_image_context:
                if pid in self.captions:
                    caption = self.captions[pid].get("caption", "")
                    if caption:
                        image_context = f"Image description: {caption}"
            
            pair = {
                "question": question_text,
                "answer": answer_text,
                "image_context": image_context,
                "problem_id": pid,
                "subject": problem.get("subject", ""),
                "topic": problem.get("topic", "")
            }
            
            pairs.append(pair)
        
        print(f"Generated {len(pairs)} question-answer pairs from {split} split")
        return pairs
    
    def generate_question_document_pairs(
        self,
        split: str = "train",
        include_image_context: bool = True
    ) -> List[Dict]:
        """
        Generate question-document pairs for retrieval training
        
        Args:
            split: Dataset split ('train', 'val', 'test')
            include_image_context: Whether to include image captions
            
        Returns:
            List of question-document pairs
        """
        if self.problems is None:
            self.load_data()
        
        split_problems = self.data_loader.get_split_problems(split)
        pairs = []
        
        # Initialize vector store to get document chunks
        if self.vector_store is None:
            self.vector_store = ScienceQAVectorStore()
            # Load all problems to build document chunks
            all_problems = self.data_loader.load_problems()
            documents = self.vector_store.load_documents_from_problems(
                all_problems, self.captions, None
            )
            # Create a mapping from problem_id to document chunks
            self.problem_documents = {}
            for doc in documents:
                pid = doc.metadata.get("problem_id")
                if pid not in self.problem_documents:
                    self.problem_documents[pid] = []
                self.problem_documents[pid].append(doc.page_content)
        
        for pid, problem in split_problems.items():
            question = problem.get("question", "")
            choices = problem.get("choices", [])
            explanation = problem.get("explanation") or problem.get("solution", "")
            
            if not question:
                continue
            
            # Build question text
            question_text = question
            if choices:
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                question_text += f"\n{choices_text}"
            
            # Get document chunks for this problem (positive documents)
            if pid in self.problem_documents:
                doc_chunks = self.problem_documents[pid]
                # Use the first chunk or combine chunks as positive document
                positive_doc = "\n\n".join(doc_chunks[:2])  # Use first 2 chunks
            else:
                # Fallback: build document from problem content
                content_parts = []
                if explanation:
                    content_parts.append(f"Explanation: {explanation}")
                if pid in self.captions:
                    caption = self.captions[pid].get("caption", "")
                    if caption:
                        content_parts.append(f"Image description: {caption}")
                positive_doc = "\n\n".join(content_parts) if content_parts else explanation or ""
            
            if not positive_doc:
                continue
            
            pair = {
                "question": question_text,
                "document": positive_doc,
                "problem_id": pid,
                "subject": problem.get("subject", ""),
                "topic": problem.get("topic", "")
            }
            
            pairs.append(pair)
        
        print(f"Generated {len(pairs)} question-document pairs from {split} split")
        return pairs
    
    def _find_hard_negatives(
        self,
        anchor_emb: np.ndarray,
        positive_emb: np.ndarray,
        candidate_texts: List[str],
        candidate_embs: np.ndarray,
        num_negatives: int
    ) -> List[str]:
        """
        Find hard negative samples using pre-computed embeddings
        
        Args:
            anchor_emb: Pre-computed anchor embedding
            positive_emb: Pre-computed positive embedding
            candidate_texts: Candidate negative texts
            candidate_embs: Pre-computed candidate embeddings
            num_negatives: Number of negatives to return
            
        Returns:
            List of hard negative texts
        """
        if len(candidate_texts) <= num_negatives:
            return candidate_texts
        
        try:
            # Find negatives that are similar to anchor but different from positive
            # Hard negatives: high similarity to anchor, low similarity to positive
            anchor_similarities = np.dot(candidate_embs, anchor_emb)
            positive_similarities = np.dot(candidate_embs, positive_emb)
            
            # Score: high anchor similarity but low positive similarity
            scores = anchor_similarities - 0.5 * positive_similarities
            
            # Select top-k hardest negatives
            top_indices = np.argsort(scores)[-num_negatives:]
            hard_negatives = [candidate_texts[i] for i in top_indices]
            
            return hard_negatives
        except Exception as e:
            print(f"⚠️ Warning: Hard negative mining failed: {e}, using random sampling")
            return random.sample(candidate_texts, min(num_negatives, len(candidate_texts)))
    
    def generate_positive_negative_pairs(
        self,
        pairs: List[Dict],
        num_negatives_per_positive: int = 4,
        task_type: str = "qa",  # "qa" for question-answer, "qd" for question-document
        max_candidates: int = 100  # Limit candidate negatives for hard mining
    ) -> List[Dict]:
        """
        Generate positive and negative pairs for contrastive learning
        Supports hard negative mining with batch optimization
        
        Args:
            pairs: List of question-answer or question-document pairs
            num_negatives_per_positive: Number of negative samples per positive pair
            task_type: "qa" for question-answer pairs, "qd" for question-document pairs
            max_candidates: Maximum number of candidate negatives to consider for hard mining
            
        Returns:
            List of training examples with positive and negative pairs
        """
        if self.use_hard_negatives:
            self._initialize_embedding_model()
        
        training_examples = []
        
        # Group by subject for better negative sampling
        by_subject = {}
        for pair in pairs:
            subject = pair.get("subject", "unknown")
            if subject not in by_subject:
                by_subject[subject] = []
            by_subject[subject].append(pair)
        
        # Determine the key for positive/negative text
        positive_key = "answer" if task_type == "qa" else "document"
        
        # Pre-compute embeddings for hard negative mining (batch optimization)
        if self.use_hard_negatives and self.embedding_model:
            print(f"📊 Pre-computing embeddings for hard negative mining...")
            # Collect all unique texts
            all_questions = [p["question"] for p in pairs]
            all_positives = [p[positive_key] for p in pairs]
            all_unique_texts = list(set(all_questions + all_positives))
            
            # Batch compute embeddings
            print(f"   Computing embeddings for {len(all_unique_texts)} unique texts...")
            text_to_emb = {}
            batch_size = 32  # Process in batches
            
            for i in tqdm(range(0, len(all_unique_texts), batch_size), desc="Computing embeddings"):
                batch_texts = all_unique_texts[i:i + batch_size]
                try:
                    batch_embs = self.embedding_model.embed_documents(batch_texts)
                    for text, emb in zip(batch_texts, batch_embs):
                        text_to_emb[text] = np.array(emb)
                except Exception as e:
                    print(f"⚠️ Warning: Batch embedding failed: {e}")
                    # Fallback: compute individually
                    for text in batch_texts:
                        try:
                            emb = self.embedding_model.embed_query(text)
                            text_to_emb[text] = np.array(emb)
                        except:
                            pass
            
            print(f"✅ Pre-computed {len(text_to_emb)} embeddings")
        else:
            text_to_emb = {}
        
        # Generate training examples with progress bar
        print(f"\n🔄 Generating positive-negative pairs ({task_type} task)...")
        for pair in tqdm(pairs, desc=f"Processing {task_type} pairs"):
            question = pair["question"]
            positive_text = pair[positive_key]
            
            # Positive pair: question + correct answer/document
            positive = {
                "anchor": question,
                "positive": positive_text,
                "label": 1,
                "task_type": task_type
            }
            
            # Negative pairs: question + wrong answers/documents
            negatives = []
            subject = pair.get("subject", "unknown")
            same_subject_pairs = by_subject.get(subject, pairs)
            
            # Sample negatives from same subject (hard negatives)
            candidate_texts = [p[positive_key] for p in same_subject_pairs if p[positive_key] != positive_text]
            
            if len(candidate_texts) < num_negatives_per_positive:
                # If not enough same-subject negatives, sample from all
                all_texts = [p[positive_key] for p in pairs if p[positive_key] != positive_text]
                candidate_texts.extend(all_texts)
            
            # Limit candidate size for efficiency
            if len(candidate_texts) > max_candidates:
                candidate_texts = random.sample(candidate_texts, max_candidates)
            
            # Use hard negative mining if enabled and embeddings are available
            if self.use_hard_negatives and self.embedding_model and text_to_emb:
                # Get pre-computed embeddings
                anchor_emb = text_to_emb.get(question)
                positive_emb = text_to_emb.get(positive_text)
                
                if anchor_emb is not None and positive_emb is not None:
                    # Get candidate embeddings
                    candidate_embs_list = []
                    valid_candidates = []
                    for cand_text in candidate_texts:
                        cand_emb = text_to_emb.get(cand_text)
                        if cand_emb is not None:
                            candidate_embs_list.append(cand_emb)
                            valid_candidates.append(cand_text)
                    
                    if len(candidate_embs_list) >= num_negatives_per_positive:
                        candidate_embs_array = np.array(candidate_embs_list)
                        sampled_negatives = self._find_hard_negatives(
                            anchor_emb, positive_emb, valid_candidates, candidate_embs_array, num_negatives_per_positive
                        )
                    else:
                        # Fallback to random if not enough embeddings
                        sampled_negatives = random.sample(
                            candidate_texts,
                            min(num_negatives_per_positive, len(candidate_texts))
                        )
                else:
                    # Fallback to random if embeddings not available
                    sampled_negatives = random.sample(
                        candidate_texts,
                        min(num_negatives_per_positive, len(candidate_texts))
                    )
            else:
                # Random sampling
                sampled_negatives = random.sample(
                    candidate_texts,
                    min(num_negatives_per_positive, len(candidate_texts))
                )
            
            for neg_text in sampled_negatives:
                negatives.append({
                    "anchor": question,
                    "negative": neg_text,
                    "label": 0,
                    "task_type": task_type
                })
            
            training_examples.append({
                "positive": positive,
                "negatives": negatives
            })
        
        print(f"\n✅ Generated {len(training_examples)} training examples ({task_type} task)")
        print(f"   Total pairs: {len(training_examples)} positives + {len(training_examples) * num_negatives_per_positive} negatives")
        if self.use_hard_negatives:
            print(f"   ✅ Used hard negative mining with batch optimization")
        
        return training_examples
    
    def prepare_training_data(
        self,
        split: str = "train",
        output_path: Optional[Path] = None,
        num_negatives: int = 4,
        include_image_context: bool = True,
        multi_task: bool = True,
        qa_weight: float = 0.5,
        qd_weight: float = 0.5
    ) -> Path:
        """
        Prepare complete training dataset with multi-task learning support
        
        Args:
            split: Dataset split to use
            output_path: Output file path
            num_negatives: Number of negative samples per positive
            include_image_context: Whether to include image context
            multi_task: Whether to include both QA and QD tasks
            qa_weight: Weight for question-answer task (if multi_task=True)
            qd_weight: Weight for question-document task (if multi_task=True)
            
        Returns:
            Path to saved training data file
        """
        all_training_examples = []
        
        if multi_task:
            print("=" * 60)
            print("Preparing Multi-Task Training Data")
            print("=" * 60)
            
            # Task 1: Question-Answer pairs
            print("\n📝 Task 1: Question-Answer Similarity")
            qa_pairs = self.generate_question_answer_pairs(split, include_image_context)
            qa_examples = self.generate_positive_negative_pairs(qa_pairs, num_negatives, task_type="qa")
            
            # Add task weights
            for example in qa_examples:
                example["task_weight"] = qa_weight
            
            all_training_examples.extend(qa_examples)
            
            # Task 2: Question-Document pairs
            print("\n📚 Task 2: Question-Document Retrieval")
            qd_pairs = self.generate_question_document_pairs(split, include_image_context)
            qd_examples = self.generate_positive_negative_pairs(qd_pairs, num_negatives, task_type="qd")
            
            # Add task weights
            for example in qd_examples:
                example["task_weight"] = qd_weight
            
            all_training_examples.extend(qd_examples)
            
            print(f"\n✅ Multi-task training data prepared:")
            print(f"   QA examples: {len(qa_examples)}")
            print(f"   QD examples: {len(qd_examples)}")
            print(f"   Total examples: {len(all_training_examples)}")
            print(f"   QA weight: {qa_weight}, QD weight: {qd_weight}")
        else:
            # Single task: only QA pairs (backward compatibility)
            print("Preparing Single-Task Training Data (QA only)")
            qa_pairs = self.generate_question_answer_pairs(split, include_image_context)
            all_training_examples = self.generate_positive_negative_pairs(qa_pairs, num_negatives, task_type="qa")
            for example in all_training_examples:
                example["task_weight"] = 1.0
        
        # Shuffle examples
        random.shuffle(all_training_examples)
        
        # Save to file
        if output_path is None:
            output_path = config.TRAINING_DATA_DIR / f"jina_training_data_{split}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_training_examples, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Training data saved to: {output_path}")
        print(f"   Total examples: {len(all_training_examples)}")
        
        return output_path

