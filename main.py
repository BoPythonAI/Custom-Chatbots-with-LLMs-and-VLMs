"""
SQA Project Main Program
Custom Chatbots with LLMs - ScienceQA Dataset
"""
import sys
import argparse
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description="SQA - Custom Chatbots with LLMs")
    parser.add_argument(
        "mode",
        choices=["build_db", "process_images", "interactive"],
        help="Running mode"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process (for testing)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "build_db":
        build_vector_database()
    elif args.mode == "process_images":
        process_images_with_llava(max_images=args.max_images)
    elif args.mode == "interactive":
        answer_question_interactive()


if __name__ == "__main__":
    main()

