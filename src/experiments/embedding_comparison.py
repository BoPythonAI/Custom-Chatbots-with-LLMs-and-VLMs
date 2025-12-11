"""
Embedding Model Comparison Experiments
Compares Jina v2 (fine-tuned) vs Jina v2 (original) vs existing embeddings
"""
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.data.data_loader import ScienceQADataLoader
from src.rag.embeddings.jina_embedding import JinaEmbedding
from src.rag.vector_store import ScienceQAVectorStore


class EmbeddingComparison:
    """
    Compare different embedding models on ScienceQA dataset
    """
    
    def __init__(self):
        """Initialize comparison experiment"""
        self.data_loader = ScienceQADataLoader()
        self.results = {}
    
    def _load_single_embedding(self, model_key: str):
        """
        Load a single embedding model
        
        Args:
            model_key: Model identifier ('jina_v2_original', 'jina_v2_finetuned', 'huggingface')
            
        Returns:
            Embedding model instance or None if failed
        """
        if model_key == 'jina_v2_original':
            print("Loading original Jina v2 base model...")
            try:
                embedding = JinaEmbedding(
                    model_name=config.JINA_BASE_MODEL,
                    device="cuda" if config.NUM_GPUS > 0 else "cpu"
                )
                print("✅ Original Jina v2 loaded")
                return embedding
            except Exception as e:
                print(f"❌ Failed to load original Jina v2: {e}")
                return None
        
        elif model_key == 'jina_v2_finetuned':
            # Check multiple possible paths
            finetuned_paths = []
            if config.JINA_MODEL_PATH:
                finetuned_paths.append(Path(config.JINA_MODEL_PATH))
            # Check for best_model directory
            best_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned" / "best_model"
            if best_model_path.exists():
                finetuned_paths.append(best_model_path)
            # Check for final model directory
            final_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned"
            if final_model_path.exists() and (final_model_path / "model.safetensors").exists():
                finetuned_paths.append(final_model_path)
            
            for model_path in finetuned_paths:
                if model_path.exists():
                    print(f"Loading fine-tuned Jina v2 model from {model_path}...")
                    try:
                        embedding = JinaEmbedding(
                            model_path=str(model_path),
                            device="cuda" if config.NUM_GPUS > 0 else "cpu"
                        )
                        print(f"✅ Fine-tuned Jina v2 loaded from {model_path}")
                        return embedding
                    except Exception as e:
                        print(f"❌ Failed to load fine-tuned Jina v2 from {model_path}: {e}")
                        continue
            return None
        
        elif model_key == 'huggingface':
            print("Loading existing HuggingFace embedding...")
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embedding = HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                print("✅ HuggingFace embedding loaded")
                return embedding
            except Exception as e:
                print(f"❌ Failed to load HuggingFace embedding: {e}")
                return None
        
        else:
            print(f"❌ Unknown model key: {model_key}")
            return None
    
    def _cleanup_embedding(self, embedding_model):
        """
        Clean up embedding model and free GPU memory
        
        Args:
            embedding_model: Embedding model instance to clean up
        """
        if embedding_model is None:
            return
        
        try:
            # For JinaEmbedding models
            if isinstance(embedding_model, JinaEmbedding):
                if hasattr(embedding_model, 'model'):
                    del embedding_model.model
                if hasattr(embedding_model, 'tokenizer'):
                    del embedding_model.tokenizer
                # Clear GPU cache if using CUDA
                if hasattr(embedding_model, 'device') and embedding_model.device == 'cuda':
                    torch.cuda.empty_cache()
            
            # Delete the embedding object itself
            del embedding_model
            
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
    
    def load_embeddings(self):
        """Load different embedding models"""
        embeddings = {}
        
        # 1. Original Jina v2 base model
        print("Loading original Jina v2 base model...")
        try:
            embeddings['jina_v2_original'] = JinaEmbedding(
                model_name=config.JINA_BASE_MODEL,
                device="cuda" if config.NUM_GPUS > 0 else "cpu"
            )
            print("✅ Original Jina v2 loaded")
        except Exception as e:
            print(f"❌ Failed to load original Jina v2: {e}")
        
        # 2. Fine-tuned Jina v2 model (check multiple possible paths)
        finetuned_paths = []
        if config.JINA_MODEL_PATH:
            finetuned_paths.append(Path(config.JINA_MODEL_PATH))
        # Check for best_model directory
        best_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned" / "best_model"
        if best_model_path.exists():
            finetuned_paths.append(best_model_path)
        # Check for final model directory
        final_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned"
        if final_model_path.exists() and (final_model_path / "model.safetensors").exists():
            finetuned_paths.append(final_model_path)
        
        for model_path in finetuned_paths:
            if model_path.exists():
                print(f"Loading fine-tuned Jina v2 model from {model_path}...")
                try:
                    embeddings['jina_v2_finetuned'] = JinaEmbedding(
                        model_path=str(model_path),
                        device="cuda" if config.NUM_GPUS > 0 else "cpu"
                    )
                    print(f"✅ Fine-tuned Jina v2 loaded from {model_path}")
                    break  # Use first available model
                except Exception as e:
                    print(f"❌ Failed to load fine-tuned Jina v2 from {model_path}: {e}")
                    continue
        
        # 3. Existing embedding (HuggingFace)
        print("Loading existing HuggingFace embedding...")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings['huggingface'] = HuggingFaceEmbeddings(
                model_name=config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ HuggingFace embedding loaded")
        except Exception as e:
            print(f"❌ Failed to load HuggingFace embedding: {e}")
        
        return embeddings
    
    def evaluate_retrieval_quality(
        self,
        embedding_model,
        model_name: str,
        test_split: str = "test",
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate retrieval quality for an embedding model
        
        Args:
            embedding_model: Embedding model instance
            model_name: Name of the model
            test_split: Test split to use
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating {model_name}...")
        
        # Clear GPU cache before building vector store
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load test data
        test_problems = self.data_loader.get_split_problems(test_split)
        problems = self.data_loader.load_problems()
        captions = self.data_loader.load_captions()
        
        # Build vector store with this embedding
        from langchain_core.documents import Document
        from langchain_community.vectorstores import Chroma
        
        documents = []
        for pid, problem in problems.items():
            content_parts = []
            question = problem.get("question", "")
            if question:
                content_parts.append(f"Question: {question}")
            
            choices = problem.get("choices", [])
            if choices:
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                content_parts.append(f"Choices:\n{choices_text}")
            
            explanation = problem.get("explanation") or problem.get("solution", "")
            if explanation:
                content_parts.append(f"Explanation: {explanation}")
            
            if pid in captions:
                caption = captions[pid].get("caption", "")
                if caption:
                    content_parts.append(f"Image description: {caption}")
            
            content = "\n\n".join(content_parts)
            
            metadata = {
                "problem_id": pid,
                "subject": problem.get("subject", ""),
                "answer": problem.get("answer", "")
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=str(config.VECTOR_DB_DIR / f"comparison_{model_name}")
        )
        
        # Evaluate retrieval
        correct_retrievals = 0
        total_questions = 0
        reciprocal_ranks = []
        
        for pid, problem in tqdm(test_problems.items(), desc=f"Evaluating {model_name}"):
            question = problem.get("question", "")
            correct_answer = problem.get("answer", "")
            
            if not question:
                continue
            
            # Retrieve similar documents
            retrieved_docs = vector_store.similarity_search(question, k=top_k)
            
            # Check if correct answer is in retrieved documents
            found = False
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc.metadata.get("problem_id") == pid:
                    found = True
                    reciprocal_ranks.append(1.0 / rank)
                    break
            
            if found:
                correct_retrievals += 1
            else:
                reciprocal_ranks.append(0.0)
            
            total_questions += 1
        
        # Calculate metrics
        recall_at_k = correct_retrievals / total_questions if total_questions > 0 else 0
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        
        results = {
            "model_name": model_name,
            "total_questions": total_questions,
            "correct_retrievals": correct_retrievals,
            "recall_at_k": recall_at_k,
            "mrr": mrr,
            "top_k": top_k
        }
        
        print(f"  Recall@{top_k}: {recall_at_k:.4f}")
        print(f"  MRR: {mrr:.4f}")
        
        return results
    
    def run_comparison(
        self,
        test_split: str = "test",
        top_k: int = 5,
        output_path: Optional[Path] = None
    ) -> Dict:
        """
        Run comparison experiment (retrieval quality only)
        Models are loaded and evaluated one at a time to save GPU memory
        
        Args:
            test_split: Test split to use
            top_k: Number of top results
            output_path: Path to save results
            
        Returns:
            Comparison results dictionary
        """
        print("=" * 60)
        print("Embedding Model Comparison Experiment")
        print("=" * 60)
        print("💡 Using memory-efficient mode: loading models one at a time")
        print("=" * 60)
        
        # Define model configurations to evaluate (check availability without loading)
        model_configs = []
        
        # Check which models are available
        # Jina v2 original - always available
        model_configs.append('jina_v2_original')
        
        # Jina v2 finetuned - check if exists
        finetuned_paths = []
        if config.JINA_MODEL_PATH:
            finetuned_paths.append(Path(config.JINA_MODEL_PATH))
        best_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned" / "best_model"
        if best_model_path.exists():
            finetuned_paths.append(best_model_path)
        final_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned"
        if final_model_path.exists() and (final_model_path / "model.safetensors").exists():
            finetuned_paths.append(final_model_path)
        
        if finetuned_paths:
            model_configs.append('jina_v2_finetuned')
        
        # HuggingFace - always available
        model_configs.append('huggingface')
        
        print(f"\n✅ Will evaluate {len(model_configs)} models: {', '.join(model_configs)}")
        
        # Evaluate each model
        all_results = []
        
        for model_key in model_configs:
            embedding_model = None
            try:
                # Clear GPU cache before loading new model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load single model
                embedding_model = self._load_single_embedding(model_key)
                if embedding_model is None:
                    print(f"⚠️ Skipping {model_key}: failed to load")
                    continue
                
                # Clear GPU cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Evaluate retrieval quality
                results = self.evaluate_retrieval_quality(
                    embedding_model,
                    model_key,
                    test_split,
                    top_k
                )
                all_results.append(results)
                
            except Exception as e:
                print(f"❌ Error evaluating {model_key}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up model immediately after evaluation
                print(f"🧹 Cleaning up {model_key}...")
                self._cleanup_embedding(embedding_model)
                embedding_model = None
                # Final cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Save results
        comparison_results = {
            "test_split": test_split,
            "top_k": top_k,
            "models": all_results
        }
        
        if output_path is None:
            output_path = config.EXPERIMENT_OUTPUT_DIR / "embedding_comparison.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Comparison results saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        for result in all_results:
            print(f"{result['model_name']:30s} | Recall@{top_k}: {result['recall_at_k']:.4f} | MRR: {result['mrr']:.4f}")
        
        return comparison_results
    
    def evaluate_answer_quality(
        self,
        embedding_model,
        model_name: str,
        test_split: str = "test",
        top_k: int = 5
    ) -> Dict:
        """
        Evaluate answer generation quality for an embedding model
        
        Args:
            embedding_model: Embedding model instance
            model_name: Name of the model
            test_split: Test split to use
            top_k: Number of top results to retrieve
            
        Returns:
            Dictionary with answer quality metrics
        """
        from src.evaluation.answer_metrics import AnswerEvaluator
        from src.rag.rag_system import ScienceQARAGSystem
        from src.rag.vector_store import ScienceQAVectorStore
        from src.llm.qwen_model import QwenLLM
        
        print(f"\nEvaluating answer quality for {model_name}...")
        
        # Clear GPU cache before building vector store
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load test data
        test_problems = self.data_loader.get_split_problems(test_split)
        problems = self.data_loader.load_problems()
        captions = self.data_loader.load_captions()
        
        # Build vector store with this embedding
        # Use a temporary directory for comparison
        temp_vector_db_dir = config.VECTOR_DB_DIR / f"answer_eval_{model_name}"
        vector_store = ScienceQAVectorStore(
            persist_directory=str(temp_vector_db_dir),
            embedding_model=embedding_model
        )
        
        # Load documents and build vector store
        documents = vector_store.load_documents_from_problems(problems, captions)
        vector_store.build_vector_store(documents, collection_name=f"scienceqa")
        
        # Load the vector store
        vector_store.load_vector_store(collection_name="scienceqa")
        
        # Initialize RAG system
        llm = QwenLLM()
        rag_system = ScienceQARAGSystem(
            vector_store=vector_store,
            llm=llm,
            problems=problems,
            captions=captions
        )
        
        # Generate answers
        generated_answers = []
        ground_truth_indices = []
        choices_list = []
        reference_texts = []
        
        print(f"Generating answers for {len(test_problems)} test questions...")
        for pid, problem in tqdm(test_problems.items(), desc=f"Generating answers ({model_name})"):
            question = problem.get("question", "")
            choices = problem.get("choices", [])
            correct_idx = problem.get("answer")
            solution = problem.get("solution", "")
            
            if not question:
                continue
            
            try:
                # Generate answer using RAG
                result = rag_system.answer_with_rag(
                    question=question,
                    choices=choices,
                    problem_id=pid,
                    k=top_k
                )
                
                generated_answers.append(result["answer"])
                ground_truth_indices.append(correct_idx)
                choices_list.append(choices)
                
                # Get reference text (choice text + solution)
                if correct_idx is not None and correct_idx < len(choices):
                    ref_text = choices[correct_idx]
                    if solution:
                        ref_text += " " + solution
                    reference_texts.append(ref_text)
                else:
                    reference_texts.append("")
            except Exception as e:
                print(f"Warning: Failed to generate answer for problem {pid}: {e}")
                generated_answers.append("")
                ground_truth_indices.append(correct_idx if correct_idx is not None else -1)
                choices_list.append(choices)
                reference_texts.append("")
        
        # Evaluate
        evaluator = AnswerEvaluator()
        metrics = evaluator.evaluate_all(
            generated_answers,
            ground_truth_indices,
            choices_list,
            reference_texts if reference_texts else None
        )
        
        results = {
            "model_name": model_name,
            "total_questions": len(generated_answers),
            **metrics
        }
        
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        if 'bleu' in metrics:
            print(f"  BLEU: {metrics.get('bleu', 0):.4f}")
        if 'rougeL' in metrics:
            print(f"  ROUGE-L: {metrics.get('rougeL', 0):.4f}")
        if 'bertscore_f1' in metrics:
            print(f"  BERTScore F1: {metrics.get('bertscore_f1', 0):.4f}")
        
        return results
    
    def run_full_comparison(
        self,
        test_split: str = "test",
        top_k: int = 5,
        output_path: Optional[Path] = None,
        evaluate_answers: bool = True,
        models: Optional[List[str]] = None
    ) -> Dict:
        """
        Run full comparison experiment (retrieval + answer quality)
        Models are loaded and evaluated one at a time to save GPU memory
        
        Args:
            test_split: Test split to use
            top_k: Number of top results
            output_path: Path to save results
            evaluate_answers: Whether to evaluate answer quality
            models: List of model names to evaluate (None means evaluate all available models)
                    Options: ['jina_v2_original', 'jina_v2_finetuned', 'huggingface']
            
        Returns:
            Combined comparison results dictionary
        """
        print("=" * 60)
        print("Full Embedding Model Comparison Experiment")
        print("=" * 60)
        print("💡 Using memory-efficient mode: loading models one at a time")
        print("=" * 60)
        
        # Define all available models
        all_available_models = []
        
        # Check which models are available
        # Jina v2 original - always available
        all_available_models.append('jina_v2_original')
        
        # Jina v2 finetuned - check if exists
        finetuned_paths = []
        if config.JINA_MODEL_PATH:
            finetuned_paths.append(Path(config.JINA_MODEL_PATH))
        best_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned" / "best_model"
        if best_model_path.exists():
            finetuned_paths.append(best_model_path)
        final_model_path = config.TRAINING_OUTPUT_DIR / "jina_finetuned"
        if final_model_path.exists() and (final_model_path / "model.safetensors").exists():
            finetuned_paths.append(final_model_path)
        
        if finetuned_paths:
            all_available_models.append('jina_v2_finetuned')
        
        # HuggingFace - always available
        all_available_models.append('huggingface')
        
        # Filter models based on user selection
        if models is None:
            model_configs = all_available_models
        else:
            # Validate and filter requested models
            model_configs = []
            for model in models:
                if model in all_available_models:
                    model_configs.append(model)
                else:
                    print(f"⚠️ Warning: Model '{model}' not available. Skipping.")
                    print(f"   Available models: {', '.join(all_available_models)}")
            
            if not model_configs:
                raise ValueError(f"No valid models to evaluate. Requested: {models}, Available: {all_available_models}")
        
        print(f"\n✅ Will evaluate {len(model_configs)} model(s): {', '.join(model_configs)}")
        
        # 1. Evaluate retrieval quality
        print("\n" + "=" * 60)
        print("Step 1: Retrieval Quality Evaluation")
        print("=" * 60)
        
        retrieval_results = []
        for model_key in model_configs:
            embedding_model = None
            try:
                # Clear GPU cache before loading new model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Load single model
                embedding_model = self._load_single_embedding(model_key)
                if embedding_model is None:
                    print(f"⚠️ Skipping {model_key}: failed to load")
                    continue
                
                # Clear GPU cache before evaluation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Evaluate retrieval quality
                results = self.evaluate_retrieval_quality(
                    embedding_model,
                    model_key,
                    test_split,
                    top_k
                )
                retrieval_results.append(results)
                
            except Exception as e:
                print(f"❌ Error evaluating retrieval for {model_key}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up model immediately after evaluation
                print(f"🧹 Cleaning up {model_key}...")
                self._cleanup_embedding(embedding_model)
                embedding_model = None
                # Final cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # 2. Evaluate answer quality (if requested)
        answer_results = []
        if evaluate_answers:
            print("\n" + "=" * 60)
            print("Step 2: Answer Quality Evaluation")
            print("=" * 60)
            
            for model_key in model_configs:
                embedding_model = None
                try:
                    # Clear GPU cache before loading new model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Load single model
                    embedding_model = self._load_single_embedding(model_key)
                    if embedding_model is None:
                        print(f"⚠️ Skipping {model_key}: failed to load")
                        continue
                    
                    # Clear GPU cache before evaluation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Evaluate answer quality
                    results = self.evaluate_answer_quality(
                        embedding_model,
                        model_key,
                        test_split,
                        top_k
                    )
                    answer_results.append(results)
                    
                except Exception as e:
                    print(f"❌ Error evaluating answers for {model_key}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    # Clean up model immediately after evaluation
                    print(f"🧹 Cleaning up {model_key}...")
                    self._cleanup_embedding(embedding_model)
                    embedding_model = None
                    # Final cleanup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Combine results
        combined_results = {
            "test_split": test_split,
            "top_k": top_k,
            "retrieval_quality": {
                "models": retrieval_results
            }
        }
        
        if answer_results:
            combined_results["answer_quality"] = {
                "models": answer_results
            }
        
        # Save results
        if output_path is None:
            output_path = config.EXPERIMENT_OUTPUT_DIR / "full_comparison.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(combined_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Full comparison results saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("Retrieval Quality Summary")
        print("=" * 60)
        for result in retrieval_results:
            print(f"{result['model_name']:30s} | Recall@{top_k}: {result['recall_at_k']:.4f} | MRR: {result['mrr']:.4f}")
        
        if answer_results:
            print("\n" + "=" * 60)
            print("Answer Quality Summary")
            print("=" * 60)
            for result in answer_results:
                print(f"{result['model_name']:30s} | Accuracy: {result.get('accuracy', 0):.4f} | "
                      f"BLEU: {result.get('bleu', 0):.4f} | ROUGE-L: {result.get('rougeL', 0):.4f} | "
                      f"BERTScore F1: {result.get('bertscore_f1', 0):.4f}")
        
        return combined_results

