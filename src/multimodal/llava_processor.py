"""
LLaVA multimodal image processing module
Responsible for generating scientific descriptions of images
"""
import os
import sys
from pathlib import Path
from typing import Optional, Dict
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config


class LLaVAImageProcessor:
    """
    LLaVA image processor
    Uses LLaVA-1.5-7B model to generate scientific descriptions of images
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize LLaVA image processor
        
        Args:
            model_name: HuggingFace model name
            device: Device (cuda/cpu)
        """
        # Set HuggingFace mirror and cache directory
        if os.getenv("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = os.getenv("HF_ENDPOINT")
        else:
            os.environ["HF_ENDPOINT"] = config.HF_ENDPOINT
        
        # Ensure cache directory is on data disk
        hf_home = str(config.MODEL_DIR.parent / ".hf_cache")
        os.environ["HF_HOME"] = hf_home
        os.environ["TRANSFORMERS_CACHE"] = str(config.MODEL_DIR)
        Path(hf_home).mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name or config.LLAVA_MODEL
        
        # Device configuration
        if device is None:
            try:
                if torch.cuda.is_available() and config.NUM_GPUS > 0:
                    test_tensor = torch.tensor([1.0], device="cuda")
                    del test_tensor
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except Exception as e:
                print(f"⚠️ CUDA not available, using CPU: {e}")
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🚀 Loading LLaVA model: {self.model_name}")
        print(f"💻 Using device: {self.device}")
        
        # Load model
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir=str(config.MODEL_DIR)
            )
            
            use_device_map = self.device == "cuda"
            model_kwargs = {
                "trust_remote_code": True,
                "dtype": torch.float16 if use_device_map else torch.float32,
                "low_cpu_mem_usage": True,
                "cache_dir": str(config.MODEL_DIR)
            }
            if use_device_map:
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = None
            
            # Load LLaVA model using LlavaForConditionalGeneration
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            if self.device == "cpu" or not use_device_map:
                if not use_device_map:
                    self.model = self.model.to("cpu")
            self.model.eval()
            
            print("✅ LLaVA model loaded successfully")
        except Exception as e:
            print(f"❌ LLaVA model loading failed: {e}")
            raise
    
    def generate_scientific_description(
        self, 
        image: Image.Image, 
        question_context: Optional[str] = None
    ) -> str:
        """
        Generate scientific description of image
        
        Args:
            image: PIL Image object
            question_context: Question context (optional)
            
        Returns:
            Image description text
        """
        prompt = "Please describe in detail the scientific content in this image, including objects, phenomena, processes, etc."
        
        if question_context:
            prompt += f"\n\nQuestion context: {question_context}\n\nPlease generate a more relevant scientific description based on the question context."
        
        try:
            # LLaVA uses conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            prompt_text = self.processor.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=prompt_text,
                images=[image],
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    num_beams=1
                )
            
            # Decode output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return output_text.strip()
        except Exception as e:
            print(f"⚠️ Failed to generate description: {e}")
            return ""
    
    def merge_captions(self, caption_gt: str, caption_llava: str) -> str:
        """
        Merge official caption and LLaVA caption
        
        Args:
            caption_gt: Official caption
            caption_llava: LLaVA-generated caption
            
        Returns:
            Merged caption
        """
        parts = []
        if caption_gt:
            parts.append(f"[Dataset Caption]:\n{caption_gt}")
        if caption_llava:
            parts.append(f"[LLaVA Caption]:\n{caption_llava}")
        
        return "\n\n".join(parts)

