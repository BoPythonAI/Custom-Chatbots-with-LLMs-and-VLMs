"""
SQA Project Main Program
Custom Chatbots with LLMs - ScienceQA Dataset
"""
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import json

sys.path.insert(0, str(Path(__file__).parent))
import config
from src.data.data_loader import ScienceQADataLoader
from src.multimodal.llava_processor import LLaVAImageProcessor
from src.llm.qwen_model import QwenLLM
from src.rag.vector_store import ScienceQAVectorStore
from src.rag.rag_system import ScienceQARAGSystem


def build_vector_database():
    """Build vector database"""
    print("=" * 60)
    print("Step 1: Loading data")
    print("=" * 60)
    
    loader = ScienceQADataLoader()
    problems = loader.load_problems()
    captions = loader.load_captions()
    
    print(f"Loaded {len(problems)} problems")
    print(f"Loaded {len(captions)} image captions")
    
    # Load LLaVA-generated descriptions (if exists)
    llava_captions_path = config.DATA_DIR / "llava_captions.json"
    llava_captions = None
    if llava_captions_path.exists():
        print(f"Loading LLaVA descriptions: {llava_captions_path}")
        with open(llava_captions_path, 'r', encoding='utf-8') as f:
            llava_captions = json.load(f)
    
    print("\n" + "=" * 60)
    print("Step 2: Building vector database")
    print("=" * 60)
    
    vector_store = ScienceQAVectorStore()
    documents = vector_store.load_documents_from_problems(
        problems, captions, llava_captions
    )
    vector_store.build_vector_store(documents)
    
    print("\nVector database construction completed!")
    return vector_store


def process_images_with_llava(max_images=None):
    """
    Process images with LLaVA and generate descriptions
    
    Args:
        max_images: Maximum number of images to process, None means process all
    """
    print("=" * 60)
    print("Processing images with LLaVA")
    print("=" * 60)
    
    loader = ScienceQADataLoader()
    problems = loader.load_problems()
    captions = loader.load_captions()
    
    processor = LLaVAImageProcessor()
    
    # Find all problems with images
    image_problems = {pid: prob for pid, prob in problems.items() 
                     if "image" in prob and prob["image"]}
    
    if max_images:
        image_problems = dict(list(image_problems.items())[:max_images])
    
    print(f"Found {len(image_problems)} image problems")
    
    llava_captions = {}
    
    for pid, problem in image_problems.items():
        image_name = problem["image"]
        image_path = config.IMAGE_DIR / image_name
        
        if not image_path.exists():
            for split in ["train", "val", "test"]:
                alt_path = config.IMAGE_DIR / split / image_name
                if alt_path.exists():
                    image_path = alt_path
                    break
            else:
                print(f"⚠️ Image not found: {image_name}")
                continue
        
        try:
            from PIL import Image
            image = Image.open(image_path)
            
            question = problem.get("question", "")
            choices = problem.get("choices", [])
            question_context = f"Question: {question}\nChoices: {', '.join(choices)}"
            
            # Generate LLaVA description
            llava_desc = processor.generate_scientific_description(image, question_context)
            
            # Get official caption
            official_caption = captions.get(pid, {}).get("caption", "")
            
            # Merge captions
            merged_caption = processor.merge_captions(official_caption, llava_desc)
            
            llava_captions[pid] = {
                "official_caption": official_caption,
                "llava_description": llava_desc,
                "merged_caption": merged_caption
            }
            
            print(f"✅ Processed: {pid}")
            
        except Exception as e:
            print(f"❌ Processing failed {pid}: {e}")
            continue
    
    # Save results
    output_path = config.DATA_DIR / "llava_captions.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(llava_captions, f, ensure_ascii=False, indent=2)
    
    print(f"\nProcessing completed! Processed {len(llava_captions)} images")
    print(f"Results saved to: {output_path}")
    
    return llava_captions


def answer_question_interactive():
    """Interactive Q&A"""
    print("=" * 60)
    print("Interactive Q&A System")
    print("=" * 60)
    
    # Load data
    loader = ScienceQADataLoader()
    problems = loader.load_problems()
    captions = loader.load_captions()
    
    # Load LLaVA-generated descriptions (if exists)
    llava_captions = None
    llava_captions_path = config.DATA_DIR / "llava_captions.json"
    if llava_captions_path.exists():
        print(f"Loading LLaVA descriptions: {llava_captions_path}")
        with open(llava_captions_path, 'r', encoding='utf-8') as f:
            llava_captions = json.load(f)
    
    # Load vector database
    vector_store = ScienceQAVectorStore()
    try:
        vector_store.load_vector_store()
    except FileNotFoundError:
        print("Vector database not found, building...")
        vector_store = build_vector_database()
    
    # Initialize RAG system
    llm = QwenLLM()
    rag_system = ScienceQARAGSystem(
        vector_store, 
        llm,
        problems=problems,
        captions=captions,
        llava_captions=llava_captions
    )
    
    print("\nSystem ready! Enter questions to start Q&A (type 'quit' to exit)\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        print("\nThinking...")
        result = rag_system.answer_with_rag(question=question)
        
        print("\n" + "-" * 60)
        print("Answer:")
        print(result["answer"])
        print(f"\nRetrieved {result['retrieved_documents']} relevant documents")
        print("-" * 60 + "\n")


def prepare_training_data(multi_task: bool = True, use_hard_negatives: bool = True):
    """Prepare training data for Jina fine-tuning
    
    Args:
        multi_task: Whether to use multi-task learning (QA + QD)
        use_hard_negatives: Whether to use hard negative mining
    """
    from src.training.data_preparation import TrainingDataPreparation
    
    preparator = TrainingDataPreparation(use_hard_negatives=use_hard_negatives)
    
    # Prepare training data
    print("Preparing training data...")
    preparator.prepare_training_data(
        split="train",
        multi_task=multi_task,
        qa_weight=0.5,
        qd_weight=0.5
    )
    
    # Prepare validation data
    print("\nPreparing validation data...")
    try:
        preparator.prepare_training_data(
            split="val",
            multi_task=multi_task,
            qa_weight=0.5,
            qd_weight=0.5
        )
        print("\n✅ Training and validation data preparation completed!")
        if multi_task:
            print("   Using multi-task learning: QA similarity + QD retrieval")
        if use_hard_negatives:
            print("   Using hard negative mining for better negative samples")
    except Exception as e:
        print(f"\n⚠️ Warning: Could not prepare validation data: {e}")
        print("   Training data preparation completed, but validation data is missing.")
        print("   You can train without validation, but validation is recommended for:")
        print("   - Model selection (best checkpoint)")
        print("   - Early stopping")
        print("   - Monitoring overfitting")


def train_jina_model(args):
    """Train Jina v2 embedding model"""
    import sys
    from pathlib import Path
    
    # Import training script
    train_script = Path(__file__).parent / "scripts" / "train_jina.py"
    if not train_script.exists():
        print("Error: Training script not found. Please use scripts/train_jina.py directly.")
        return
    
    # Run training script
    import subprocess
    training_data = config.TRAINING_DATA_DIR / "jina_training_data_train.json"
    
    if not training_data.exists():
        print(f"Error: Training data not found: {training_data}")
        print("Please run 'python main.py prepare_data' first.")
        return
    
    # Check for validation data (auto-use if exists)
    eval_data = None
    if hasattr(args, 'eval_data') and args.eval_data:
        eval_data = Path(args.eval_data)
    else:
        # Auto-detect validation data
        val_data = config.TRAINING_DATA_DIR / "jina_training_data_val.json"
        if val_data.exists():
            eval_data = val_data
            print(f"✅ Auto-detected validation data: {eval_data}")
        else:
            print("⚠️ Warning: No validation data found. Training without validation.")
            print(f"   To use validation set, run: python main.py prepare_data --split val")
    
    # Build command with all training parameters
    cmd = [
        sys.executable,
        str(train_script),
        "--data", str(training_data),
    ]
    
    # Add optional parameters if provided
    if hasattr(args, 'output') and args.output:
        cmd.extend(["--output", str(args.output)])
    else:
        cmd.extend(["--output", str(config.TRAINING_OUTPUT_DIR / "jina_finetuned")])
    
    if hasattr(args, 'batch_size') and args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    
    if hasattr(args, 'epochs') and args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
    
    if hasattr(args, 'gradient_accumulation_steps') and args.gradient_accumulation_steps:
        cmd.extend(["--gradient-accumulation-steps", str(args.gradient_accumulation_steps)])
    
    if hasattr(args, 'max_length') and args.max_length:
        cmd.extend(["--max-length", str(args.max_length)])
    
    if hasattr(args, 'save_steps') and args.save_steps:
        cmd.extend(["--save-steps", str(args.save_steps)])
    
    if eval_data:
        cmd.extend(["--eval-data", str(eval_data)])
    
    subprocess.run(cmd)


def compare_embeddings(evaluate_answers: bool = True, split: str = "test", top_k: int = 5, models: Optional[List[str]] = None):
    """Compare different embedding models
    
    Args:
        evaluate_answers: Whether to evaluate answer quality
        split: Dataset split to use (train/val/test)
        top_k: Number of top results to retrieve
        models: List of model names to evaluate (None means evaluate all available models)
                Options: ['jina_v2_original', 'jina_v2_finetuned', 'huggingface']
    """
    from src.experiments.embedding_comparison import EmbeddingComparison
    
    print(f"Running full embedding comparison experiment on {split} set...")
    comparison = EmbeddingComparison()
    results = comparison.run_full_comparison(
        test_split=split, 
        top_k=top_k,
        evaluate_answers=evaluate_answers,
        models=models
    )
    print("\n✅ Full comparison experiment completed!")


def main():
    parser = argparse.ArgumentParser(description="SQA - Custom Chatbots with LLMs")
    parser.add_argument(
        "mode",
        choices=["build_db", "process_images", "interactive", "prepare_data", "train_jina", "compare_embeddings"],
        help="Running mode"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )
    
    # Training parameters (only used when mode is train_jina)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size for training (default: {config.TRAINING_BATCH_SIZE})"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {config.TRAINING_EPOCHS})"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Learning rate (default: {config.TRAINING_LEARNING_RATE})"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help=f"Gradient accumulation steps (default: {config.TRAINING_GRADIENT_ACCUMULATION_STEPS})"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help=f"Maximum sequence length (default: {config.TRAINING_MAX_LENGTH})"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help=f"Steps between checkpoints (default: {config.TRAINING_SAVE_STEPS})"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data JSON file (optional)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for trained model (default: training_output/jina_finetuned)"
    )
    
    # Comparison experiment parameters
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split for comparison (default: test)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to retrieve (default: 5)"
    )
    parser.add_argument(
        "--no-answer-eval",
        action="store_true",
        help="Skip answer quality evaluation (only evaluate retrieval quality)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["jina_v2_original", "jina_v2_finetuned", "huggingface"],
        help="Specific models to evaluate (default: all available models). "
             "Options: jina_v2_original, jina_v2_finetuned, huggingface"
    )
    
    args = parser.parse_args()
    
    if args.mode == "build_db":
        build_vector_database()
    elif args.mode == "process_images":
        process_images_with_llava(max_images=args.max_images)
    elif args.mode == "interactive":
        answer_question_interactive()
    elif args.mode == "prepare_data":
        prepare_training_data()
    elif args.mode == "train_jina":
        train_jina_model(args)
    elif args.mode == "compare_embeddings":
        compare_embeddings(
            evaluate_answers=not args.no_answer_eval,
            split=args.split,
            top_k=args.top_k,
            models=args.models
        )


if __name__ == "__main__":
    main()

