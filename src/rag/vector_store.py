"""
Vector Database integration module
Build vector database using LangChain and ChromaDB
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
except ImportError:
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class ScienceQAVectorStore:
    """ScienceQA vector database management class"""
    
    def __init__(self, persist_directory: Optional[str] = None):
        """
        Initialize vector database
        
        Args:
            persist_directory: Persistence directory path
        """
        self.persist_directory = persist_directory or config.VECTOR_DB_PATH
        
        # Initialize embedding model
        device = 'cpu'  # Use CPU to avoid CUDA compatibility issues
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True},
            cache_folder=str(config.MODEL_DIR)
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )
        
        self.vectorstore = None
    
    def load_documents_from_problems(
        self, 
        problems: Dict,
        captions: Dict,
        llava_captions: Optional[Dict] = None
    ) -> List[Document]:
        """
        Load documents from problems.json
        
        Args:
            problems: problems.json content
            captions: captions.json content
            llava_captions: LLaVA-generated descriptions (optional)
            
        Returns:
            Document list
        """
        documents = []
        
        for problem_id, problem in problems.items():
            content_parts = []
            
            # Add question
            question = problem.get("question", "")
            if question:
                content_parts.append(f"Question: {question}")
            
            # Add choices
            choices = problem.get("choices", [])
            if choices:
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                content_parts.append(f"Choices:\n{choices_text}")
            
            # Add explanation/solution
            explanation = problem.get("explanation") or problem.get("solution") or problem.get("lecture", "")
            if explanation:
                content_parts.append(f"Explanation: {explanation}")
            
            # Add hint
            hint = problem.get("hint", "")
            if hint:
                content_parts.append(f"Hint: {hint}")
            
            # Add official caption
            if problem_id in captions:
                official_caption = captions[problem_id].get("caption", "")
                if official_caption:
                    content_parts.append(f"Image description: {official_caption}")
            
            # Add LLaVA-generated description (if exists)
            if llava_captions and problem_id in llava_captions:
                merged_caption = llava_captions[problem_id].get("merged_caption", "")
                if merged_caption:
                    content_parts.append(f"Merged image description: {merged_caption}")
            
            content = "\n\n".join(content_parts)
            
            # Create metadata
            metadata = {
                "problem_id": problem_id,
                "subject": problem.get("subject", ""),
                "topic": problem.get("topic", ""),
                "grade": problem.get("grade", ""),
                "has_image": "image" in problem and bool(problem["image"]),
                "answer": problem.get("answer", "")
            }
            
            # Split documents
            texts = self.text_splitter.split_text(content)
            
            for i, text in enumerate(texts):
                doc = Document(
                    page_content=text,
                    metadata={**metadata, "chunk_id": i}
                )
                documents.append(doc)
        
        return documents
    
    def build_vector_store(
        self,
        documents: List[Document],
        collection_name: str = "scienceqa"
    ):
        """
        Build vector database
        
        Args:
            documents: Document list
            collection_name: Collection name
        """
        print(f"📊 Building vector database, {len(documents)} document chunks...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=collection_name
        )
        
        print(f"✅ Vector database construction completed, saved to: {self.persist_directory}")
    
    def load_vector_store(self, collection_name: str = "scienceqa"):
        """
        Load existing vector database
        
        Args:
            collection_name: Collection name
        """
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
            collection_name=collection_name
        )
        print(f"✅ Vector database loaded successfully")
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> List[Document]:
        """
        Similarity search
        
        Args:
            query: Query text
            k: Number of documents to return
            filter_dict: Filter conditions
            
        Returns:
            List of relevant documents
        """
        if self.vectorstore is None:
            raise ValueError("Vector database not loaded, please call load_vector_store() first")
        
        if filter_dict:
            return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            return self.vectorstore.similarity_search(query, k=k)

