"""
Prepare training data for Jina v2 embedding fine-tuning
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.training.data_preparation import TrainingDataPreparation


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for Jina v2 fine-tuning")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: data/training/jina_training_data_{split}.json)"
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=4,
        help="Number of negative samples per positive pair"
    )
    parser.add_argument(
        "--no-image-context",
        action="store_true",
        help="Exclude image context from training data"
    )
    
    args = parser.parse_args()
    
    # Prepare data
    preparator = TrainingDataPreparation()
    
    output_path = Path(args.output) if args.output else None
    
    preparator.prepare_training_data(
        split=args.split,
        output_path=output_path,
        num_negatives=args.num_negatives,
        include_image_context=not args.no_image_context
    )
    
    print("\n✅ Training data preparation completed!")


if __name__ == "__main__":
    main()

