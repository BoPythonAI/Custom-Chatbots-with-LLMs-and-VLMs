"""
RAG system integration module
Implements retrieval-augmented generation by combining VectorDB and Qwen LLM
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.llm.qwen_model import QwenLLM
from src.rag.vector_store import ScienceQAVectorStore
import config

# Lazy import LLaVA processor
_llava_processor = None

def get_llava_processor():
    """Get LLaVA image processor (lazy loading)"""
    global _llava_processor
    if _llava_processor is None:
        from src.multimodal.llava_processor import LLaVAImageProcessor
        _llava_processor = LLaVAImageProcessor()
    return _llava_processor


class ScienceQARAGSystem:
    """ScienceQA RAG system"""
    
    def __init__(
        self,
        vector_store: ScienceQAVectorStore,
        llm: QwenLLM,
        problems: Optional[Dict] = None,
        captions: Optional[Dict] = None,
        llava_captions: Optional[Dict] = None
    ):
        """
        Initialize RAG system
        
        Args:
            vector_store: Vector database instance
            llm: Qwen LLM instance
            problems: problems.json content
            captions: captions.json content
            llava_captions: LLaVA-generated descriptions
        """
        self.vector_store = vector_store
        self.llm = llm
        self.problems = problems
        self.captions = captions or {}
        self.llava_captions = llava_captions or {}
    
    def retrieve_context(
        self,
        query: str,
        k: int = None,
        subject_filter: Optional[str] = None
    ) -> List[Document]:
        """
        Retrieve relevant context
        
        Args:
            query: Query text
            k: Number of documents to return
            subject_filter: Subject filter
            
        Returns:
            List of relevant documents
        """
        k = k or config.TOP_K_RETRIEVAL
        
        filter_dict = {}
        if subject_filter:
            filter_dict["subject"] = subject_filter
        
        if filter_dict:
            documents = self.vector_store.similarity_search(query, k=k, filter_dict=filter_dict)
        else:
            documents = self.vector_store.similarity_search(query, k=k)
        
        return documents
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved context
        
        Args:
            documents: Document list
            
        Returns:
            Formatted context text
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            content = doc.page_content
            
            context_parts.append(f"[Relevant Document {i}]")
            if "subject" in metadata:
                context_parts.append(f"Subject: {metadata['subject']}")
            if "topic" in metadata:
                context_parts.append(f"Topic: {metadata['topic']}")
            context_parts.append(f"Content:\n{content}\n")
        
        return "\n".join(context_parts)
    
    def find_problem_by_id(self, problem_id: str) -> Optional[Dict]:
        """Find problem information by problem ID"""
        if self.problems and problem_id in self.problems:
            return self.problems[problem_id]
        return None
    
    def get_image_info_for_problem(self, problem_id: str, problem: Dict) -> Optional[Dict]:
        """Get image information for problem"""
        image_name = problem.get("image")
        if not image_name:
            return None
        
        # Build image path
        image_path = config.IMAGE_DIR / image_name
        if not image_path.exists():
            for split in ["train", "val", "test"]:
                alt_path = config.IMAGE_DIR / split / image_name
                if alt_path.exists():
                    image_path = alt_path
                    break
            else:
                return None
        
        # Get image description
        official_caption = self.captions.get(problem_id, {}).get("caption", "")
        llava_desc = None
        merged_caption = None
        
        if problem_id in self.llava_captions:
            llava_desc = self.llava_captions[problem_id].get("llava_description", "")
            merged_caption = self.llava_captions[problem_id].get("merged_caption", "")
        
        return {
            "image_path": str(image_path),
            "image_name": image_name,
            "official_caption": official_caption,
            "llava_description": llava_desc,
            "merged_caption": merged_caption
        }
    
    def answer_with_rag(
        self,
        question: str,
        choices: Optional[List[str]] = None,
        image_path: Optional[str] = None,
        problem_id: Optional[str] = None,
        subject: Optional[str] = None,
        k: int = None,
        auto_process_image: bool = True
    ) -> Dict[str, any]:
        """
        Answer question using RAG
        
        Args:
            question: Question text
            choices: Multiple choice options
            image_path: Image path (if provided)
            problem_id: Problem ID (if provided)
            subject: Subject (for filtering)
            k: Number of documents to retrieve
            auto_process_image: Whether to automatically process image
            
        Returns:
            Dictionary containing answer and retrieval information
        """
        is_image_question = False
        image_info = None
        image_description = None
        
        # 1. Try to find problem information by problem_id
        if problem_id and self.problems:
            problem = self.find_problem_by_id(problem_id)
            if problem:
                image_info = self.get_image_info_for_problem(problem_id, problem)
                if image_info:
                    is_image_question = True
                    if not image_path:
                        image_path = image_info["image_path"]
                    if not choices:
                        choices = problem.get("choices", [])
                    if not subject:
                        subject = problem.get("subject")
                    
                    # Use merged caption
                    if image_info.get("merged_caption"):
                        image_description = image_info["merged_caption"]
                    elif image_info.get("official_caption"):
                        image_description = image_info["official_caption"]
        
        # 2. If image path is directly provided, call LLaVA to process
        if image_path and not image_description and auto_process_image:
            try:
                from PIL import Image
                llava_processor = get_llava_processor()
                image = Image.open(image_path)
                question_context = f"Question: {question}\nChoices: {', '.join(choices) if choices else ''}"
                llava_desc = llava_processor.generate_scientific_description(image, question_context)
                
                # Get official caption and merge
                official_caption = ""
                if problem_id and problem_id in self.captions:
                    official_caption = self.captions[problem_id].get("caption", "")
                
                image_description = llava_processor.merge_captions(official_caption, llava_desc)
                is_image_question = True
                print(f"✅ LLaVA processed image and generated description")
            except Exception as e:
                print(f"⚠️ LLaVA image processing failed: {e}")
        
        # Build query
        query = question
        if choices:
            query += " " + " ".join(choices)
        
        # Retrieve relevant context
        retrieved_docs = self.retrieve_context(
            query,
            k=k,
            subject_filter=subject
        )
        
        # Format context
        context = self.format_context(retrieved_docs)
        
        # Generate answer using LLM
        answer = self.llm.answer_question(
            question=question,
            context=context,
            image_description=image_description,
            choices=choices
        )
        
        return {
            "answer": answer,
            "retrieved_documents": len(retrieved_docs),
            "context": context,
            "retrieved_docs": retrieved_docs,
            "image_description": image_description if image_description else None,
            "is_image_question": is_image_question,
            "image_info": image_info,
            "problem_id": problem_id
        }

