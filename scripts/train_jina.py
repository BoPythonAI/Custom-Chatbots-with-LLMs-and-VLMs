"""
Train Jina v2 embedding model
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.training.jina_trainer import JinaTrainer


def main():
    parser = argparse.ArgumentParser(description="Train Jina v2 embedding model")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for trained model (default: training_output/jina_finetuned)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=f"Number of training epochs (default: {config.TRAINING_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size (default: {config.TRAINING_BATCH_SIZE})"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Learning rate (default: {config.TRAINING_LEARNING_RATE})"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data JSON file (optional)"
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help=f"Steps between checkpoints (default: {config.TRAINING_SAVE_STEPS})"
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
        "--temperature",
        type=float,
        default=None,
        help=f"Temperature parameter for InfoNCE loss (default: {getattr(config, 'TRAINING_TEMPERATURE', 0.05)})"
    )
    parser.add_argument(
        "--no-in-batch-negatives",
        action="store_true",
        help="Disable in-batch negatives (default: enabled)"
    )
    
    args = parser.parse_args()
    
    # Initialize trainer with advanced options
    trainer = JinaTrainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        temperature=args.temperature,
        use_in_batch_negatives=not args.no_in_batch_negatives
    )
    
    # Set output directory - ensure it's on data disk
    if args.output:
        output_dir = Path(args.output)
        # If not absolute path or not on data disk, use TRAINING_OUTPUT_DIR
        if not output_dir.is_absolute() or not str(output_dir).startswith("/root/autodl-tmp/"):
            print(f"⚠️ Warning: Output path {output_dir} is not on data disk.")
            print(f"   Redirecting to data disk: {config.TRAINING_OUTPUT_DIR / output_dir.name}")
            output_dir = config.TRAINING_OUTPUT_DIR / output_dir.name
    else:
        output_dir = config.TRAINING_OUTPUT_DIR / "jina_finetuned"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Training outputs will be saved to: {output_dir}")
    
    # Training data path
    training_data_path = Path(args.data)
    if not training_data_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_data_path}")
    
    # Evaluation data path (optional)
    eval_data_path = Path(args.eval_data) if args.eval_data else None
    if eval_data_path and not eval_data_path.exists():
        print(f"Warning: Evaluation data not found: {eval_data_path}")
        eval_data_path = None
    
    # Train
    trainer.train(
        training_data_path=training_data_path,
        output_dir=output_dir,
        epochs=args.epochs,
        eval_data_path=eval_data_path,
        save_steps=args.save_steps
    )
    
    print(f"\n✅ Training completed! Model saved to: {output_dir}")


if __name__ == "__main__":
    main()

