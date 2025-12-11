# ScienceQA RAG 多模态智能问答系统

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于 ScienceQA 数据集的智能教育助手项目，整合了多模态视觉理解、检索增强生成（RAG）和大型语言模型等前沿技术。

## ✨ 项目特性

- 🎯 **多模态 RAG 系统**：整合 LLaVA 图像理解和 RAG 文本检索，支持图文混合问答
- 🚀 **Embedding 模型微调**：通过多任务学习和硬负样本挖掘优化检索性能
- 📊 **完整评估体系**：检索质量评估（Recall@K, MRR）
- 🔧 **易于使用**：提供命令行和 Web 界面两种交互方式
- 📈 **性能提升**：微调后 Recall@5 提升 4.89%，正确检索数增加 12.2%

## 🏗️ 架构说明

```
用户交互层 (Streamlit Web / CLI)
    ↓
RAG系统核心层 (ScienceQARAGSystem)
    ↓
┌─────────────┬─────────────┬─────────────┐
│  向量检索   │  多模态处理  │  LLM生成    │
│ VectorStore │   LLaVA     │  Qwen-max   │
└─────────────┴─────────────┴─────────────┘
```

- **LLaVA**：负责"看图变文字"，生成图片的科学化描述
- **VectorDB**：负责"记知识并检索"，存储和检索相关知识
- **Qwen-max**：负责"综合理解 + 推理答题"，基于检索到的上下文生成答案
- **Jina Embedding**：支持原始模型和微调模型，提升检索质量

## 🚀 快速开始

### 环境要求

- Python 3.12+
- CUDA 11.8+ (GPU 推荐)
- 至少 16GB 内存
- 至少 50GB 磁盘空间（用于模型和数据）

### 1. 克隆项目

```bash
git clone https://github.com/BoPythonAI/Custom-Chatbots-with-LLMs-and-VLMs.git
cd Custom-Chatbots-with-LLMs-and-VLMs
```

### 2. 设置环境

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 设置环境变量（创建 .env 文件）
echo "DASHSCOPE_API_KEY=your_api_key_here" > .env
```

### 3. 准备数据

下载 ScienceQA 数据集并放置在 `data/scienceqa/` 目录下：
- `problems.json`
- `captions.json`
- `pid_splits.json`
- `images/` 目录

### 4. 构建向量数据库

```bash
python main.py build_db
```

### 5. 处理图片（可选）

```bash
# 处理10张图片测试
python main.py process_images --max-images 10

# 处理所有图片
python main.py process_images
```

### 6. 启动Web界面

```bash
streamlit run streamlit_app.py
```

然后在浏览器中访问 `http://localhost:8501`

## 📖 使用指南

### 命令行交互

```bash
python main.py interactive
```

### 训练 Embedding 模型

```bash
# 1. 准备训练数据（多任务学习 + 硬负样本）
python main.py prepare_data

# 2. 训练模型
python main.py train_jina --batch-size 4 --epochs 2 --learning-rate 5e-6

# 3. 评估模型
python main.py compare_embeddings --no-answer-eval --models jina_v2_finetuned
```

### 运行模式

- `build_db`: 构建向量数据库
- `process_images`: 使用 LLaVA 处理图片
- `interactive`: 命令行交互式问答
- `prepare_data`: 准备训练数据（支持多任务学习和硬负样本）
- `train_jina`: 训练 Jina Embedding 模型
- `compare_embeddings`: 对比评估不同 embedding 模型

## 📁 项目结构

```
SQA/
├── config.py                      # 配置文件
├── main.py                        # 主程序入口
├── streamlit_app.py               # Streamlit Web界面
├── requirements.txt               # 依赖列表
├── README.md                      # 项目说明
├── FINAL_PROJECT_REPORT.md        # 完整项目报告
├── LICENSE                        # 许可证
├── .gitignore                     # Git忽略文件
│
├── src/                           # 源代码目录
│   ├── data/
│   │   └── data_loader.py        # 数据加载模块
│   ├── multimodal/
│   │   └── llava_processor.py   # LLaVA图像处理
│   ├── llm/
│   │   └── qwen_model.py         # Qwen LLM集成
│   ├── rag/
│   │   ├── embeddings/
│   │   │   └── jina_embedding.py  # Jina Embedding封装
│   │   ├── vector_store.py       # 向量存储
│   │   └── rag_system.py         # RAG系统
│   ├── training/
│   │   ├── data_preparation.py   # 训练数据准备（多任务+硬负样本）
│   │   └── jina_trainer.py      # 模型训练器（多任务学习）
│   ├── evaluation/
│   │   └── answer_metrics.py     # 答案质量评估指标
│   └── experiments/
│       └── embedding_comparison.py  # 模型对比评估（内存优化）
│
├── scripts/                       # 脚本目录
│   ├── train_jina.py             # 训练脚本
│   └── prepare_training_data.py  # 数据准备脚本
│
├── data/                          # 数据目录
│   ├── scienceqa/                # ScienceQA数据集
│   ├── training/                  # 训练数据
│   └── vectordb/                  # 向量数据库
│
├── training_output/               # 训练输出
│   └── jina_finetuned/           # 微调后的模型
│       ├── best_model/           # 最佳模型（验证损失最低）
│       └── checkpoint-*/         # 训练检查点
│
└── experiments/                   # 实验结果
    └── full_comparison.json      # 对比评估结果（JSON格式）
```

## 🎯 核心功能

### 1. 多模态 RAG 系统
- 整合 LLaVA 图像理解和 RAG 文本检索
- 支持图文混合问答
- 融合官方 caption 和 LLaVA 生成的描述

### 2. Embedding 模型微调
- **多任务学习**：同时训练 QA 相似度和 QD 检索相似度
- **硬负样本挖掘**：批量预计算，动态选择最难的负样本
- **In-Batch Negatives**：基于 SimCSE/DPR 的最佳实践
- **训练优化**：Cosine Annealing with Warmup，早停机制

### 3. 完整评估体系
- **检索质量评估**：Recall@K, MRR
- **内存优化**：顺序加载/卸载模型，避免 GPU OOM

## 📊 实验结果

### 检索质量对比（Test Set, Top-K=5）

| 模型 | Recall@5 | MRR | 正确检索数 |
|------|----------|-----|------------|
| Jina v2 Original | 0.4001 | 0.3990 | 1,697 |
| **Jina v2 Fine-tuned** | **0.4490** | **0.4028** | **1,904** |
| HuggingFace | 0.3893 | 0.3866 | 1,651 |

**性能提升**：
- Recall@5 提升 **4.89%** (0.4001 → 0.4490)
- 正确检索数增加 **12.2%** (1,697 → 1,904)

详细实验结果请参考 [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md)

## ⚙️ 配置说明

### 环境变量

创建 `.env` 文件并设置以下变量：

```bash
# 必需
DASHSCOPE_API_KEY=your_api_key_here

# 可选
QWEN_MODEL=qwen-max
LLAVA_MODEL=llava-hf/llava-1.5-7b-hf
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
JINA_BASE_MODEL=jinaai/jina-embeddings-v2-base-en
```



## 📝 注意事项

- 首次运行需要下载 LLaVA 模型（约13GB）和 embedding 模型
- Qwen API 需要有效的 DashScope API 密钥
- 训练需要 GPU 支持（推荐至少 16GB 显存）
- 数据文件较大，建议使用 SSD 存储

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [ScienceQA Dataset](https://scienceqa.github.io/)
- [LLaVA](https://llava-vl.github.io/)
- [Jina AI](https://jina.ai/)
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)

## 📚 相关文档

- [完整项目报告](FINAL_PROJECT_REPORT.md)


