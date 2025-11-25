"""
Configuration file for SQA project
All files are stored on the data disk /root/autodl-tmp
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths - all in data disk
BASE_DIR = Path("/root/autodl-tmp/SQA")
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
CACHE_DIR = BASE_DIR / ".cache"

# Create directories if they don't exist
for dir_path in [BASE_DIR, DATA_DIR, MODEL_DIR, VECTOR_DB_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API Keys
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    raise ValueError("DASHSCOPE_API_KEY environment variable is required. Please set it in .env file or environment variables.")

# HuggingFace Configuration
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com")

# Model Configuration
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-max")
LLAVA_MODEL = os.getenv("LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf")

# Vector Database Configuration
VECTOR_DB_PATH = str(VECTOR_DB_DIR)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# GPU Configuration
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0,1")
NUM_GPUS = int(os.getenv("NUM_GPUS", "2"))

# ScienceQA Dataset Paths
SCIENCEQA_DATA_DIR = DATA_DIR / "scienceqa"
PROBLEMS_JSON = SCIENCEQA_DATA_DIR / "problems.json"
CAPTIONS_JSON = SCIENCEQA_DATA_DIR / "captions.json"
PID_SPLITS_JSON = SCIENCEQA_DATA_DIR / "pid_splits.json"
IMAGE_DIR = SCIENCEQA_DATA_DIR / "images"

# RAG Configuration
TOP_K_RETRIEVAL = 5
TEMPERATURE = 0.7
MAX_TOKENS = 2048

# Logging
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

