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

# Ensure HuggingFace and PyTorch cache directories are on data disk
HF_CACHE_DIR = BASE_DIR.parent / ".hf_cache"
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TORCH_CACHE_DIR = CACHE_DIR / "torch"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_CACHE_DIR = CACHE_DIR / "datasets"
DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set environment variables to ensure all caches go to data disk
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(MODEL_DIR)
os.environ["HF_DATASETS_CACHE"] = str(DATASETS_CACHE_DIR)
os.environ["TORCH_HOME"] = str(TORCH_CACHE_DIR)

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

# Jina v2 Embedding Model Configuration
JINA_BASE_MODEL = os.getenv("JINA_BASE_MODEL", "jinaai/jina-embeddings-v2-base-en")
JINA_MODEL_PATH = os.getenv("JINA_MODEL_PATH", None)  # Path to fine-tuned model, None means use base model
JINA_MODEL_DIR = MODEL_DIR / "jina"
JINA_MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

# Training Configuration
TRAINING_DATA_DIR = DATA_DIR / "training"
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_OUTPUT_DIR = BASE_DIR / "training_output"
TRAINING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training Hyperparameters
TRAINING_BATCH_SIZE = int(os.getenv("TRAINING_BATCH_SIZE", "4"))
TRAINING_LEARNING_RATE = float(os.getenv("TRAINING_LEARNING_RATE", "5e-6"))  # Reduced to 5e-6 for more stable training and better generalization
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "2"))  # Reduced to 2 epochs to prevent overfitting
TRAINING_WARMUP_STEPS = int(os.getenv("TRAINING_WARMUP_STEPS", "100"))
TRAINING_MAX_LENGTH = int(os.getenv("TRAINING_MAX_LENGTH", "512"))
TRAINING_GRADIENT_ACCUMULATION_STEPS = int(os.getenv("TRAINING_GRADIENT_ACCUMULATION_STEPS", "8"))
TRAINING_EVAL_STEPS = int(os.getenv("TRAINING_EVAL_STEPS", "500"))
TRAINING_SAVE_STEPS = int(os.getenv("TRAINING_SAVE_STEPS", "500"))
TRAINING_EARLY_STOPPING_PATIENCE = int(os.getenv("TRAINING_EARLY_STOPPING_PATIENCE", "2"))  # Reduced from 3 for earlier stopping
TRAINING_EARLY_STOPPING_MIN_DELTA = float(os.getenv("TRAINING_EARLY_STOPPING_MIN_DELTA", "0.001"))
TRAINING_SAVE_BEST_MODEL = os.getenv("TRAINING_SAVE_BEST_MODEL", "true").lower()

# Advanced Training Parameters (based on Kaggle/Research best practices)
TRAINING_TEMPERATURE = float(os.getenv("TRAINING_TEMPERATURE", "0.05"))  # InfoNCE temperature parameter
TRAINING_USE_IN_BATCH_NEGATIVES = os.getenv("TRAINING_USE_IN_BATCH_NEGATIVES", "true").lower() == "true"  # Use in-batch negatives (SimCSE/DPR style)
TRAINING_WARMUP_RATIO = float(os.getenv("TRAINING_WARMUP_RATIO", "0.1"))  # Warmup ratio (10% of total steps) == "true"
# Learning rate decay configuration
TRAINING_LR_DECAY_FACTOR = float(os.getenv("TRAINING_LR_DECAY_FACTOR", "0.1"))  # Decay factor for learning rate scheduler

# Evaluation Configuration
EVAL_OUTPUT_DIR = BASE_DIR / "eval_output"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Experiment Configuration
EXPERIMENT_OUTPUT_DIR = BASE_DIR / "experiments"
EXPERIMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Embedding Configuration
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))  # Batch size for embedding documents (reduce if OOM)

