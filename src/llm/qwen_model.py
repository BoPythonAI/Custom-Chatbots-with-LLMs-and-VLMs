"""
Qwen model integration module
Uses Qwen-max as LLM Base-model
"""
import sys
from pathlib import Path
from typing import Optional
import dashscope
from dashscope import Generation

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class QwenLLM:
    """Qwen LLM model wrapper class"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "qwen-max"):
        """
        Initialize Qwen model
        
        Args:
            api_key: DashScope API key
            model_name: Qwen model name to use
        """
        self.api_key = api_key or config.DASHSCOPE_API_KEY
        if not self.api_key:
            raise ValueError("DashScope API key is required.")
        
        dashscope.api_key = self.api_key
        self.model_name = model_name or config.QWEN_MODEL
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> str:
        """
        Generate text
        
        Args:
            prompt: Input prompt
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters
            
        Returns:
            Generated text
        """
        try:
            response = Generation.call(
                model=self.model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            if response.status_code == 200:
                return response.output.text
            else:
                error_msg = f"Qwen API error: {response.message}"
                print(error_msg)
                return ""
        except Exception as e:
            print(f"Error calling Qwen API: {e}")
            return ""
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        image_description: Optional[str] = None,
        choices: Optional[list] = None
    ) -> str:
        """
        Answer ScienceQA question
        
        Args:
            question: Question text
            context: Context information (retrieved relevant documents)
            image_description: Image description (merged caption)
            choices: Multiple choice options list
            
        Returns:
            Answer text
        """
        prompt_parts = []
        
        # System prompt
        prompt_parts.append("You are a professional science education assistant, skilled at answering science questions. Please provide accurate and detailed answers based on the provided context information.")
        
        # Add context
        if context:
            prompt_parts.append(f"Relevant context information:\n{context}\n")
        
        # Add image description
        if image_description:
            prompt_parts.append(f"Image description:\n{image_description}\n")
        
        # Add question
        prompt_parts.append(f"Question: {question}")
        
        # Add choices (if multiple choice)
        if choices:
            choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            prompt_parts.append(f"Choices:\n{choices_text}")
        
        prompt_parts.append("\nPlease provide your answer and detailed explanation:")
        
        prompt = "\n".join(prompt_parts)
        
        return self.generate(
            prompt,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS
        )

