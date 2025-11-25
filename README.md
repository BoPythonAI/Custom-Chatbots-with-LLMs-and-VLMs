# SQA - Custom Chatbots with LLMs

基于ScienceQA数据集的自定义聊天机器人项目，使用LLaVA、LangChain、VectorDB和Qwen-max。

## 项目概述

本项目实现了：
- **LLaVA多模态模型**：处理图片问题，生成科学化描述
- **Qwen-max LLM**：使用Qwen-max作为LLM Base-model进行文本生成和问答
- **RAG系统**：集成LangChain和ChromaDB构建检索增强生成系统
- **Streamlit Web界面**：提供可视化的问答界面

## 架构说明

- **LLaVA**：负责"看图变文字"，生成图片的科学化描述
- **VectorDB**：负责"记知识并检索"，存储和检索相关知识
- **Qwen-max**：负责"综合理解 + 推理答题"，基于检索到的上下文生成答案

## 快速开始

### 1. 设置环境

```bash
cd /root/autodl-tmp/SQA
chmod +x setup_env.sh
./setup_env.sh
```

### 2. 构建向量数据库

```bash
source venv/bin/activate
python main.py build_db
```

### 3. 处理图片（可选）

```bash
# 处理10张图片测试
python main.py process_images --max-images 10

# 处理所有图片
python main.py process_images
```

### 4. 启动Web界面

```bash
./run_streamlit.sh
```

然后在浏览器中访问 `http://localhost:8501`

## 运行模式

- `build_db`: 构建向量数据库
- `process_images`: 使用LLaVA处理图片
- `interactive`: 命令行交互式问答

## 项目结构

```
SQA/
├── config.py                 # 配置文件
├── main.py                   # 主程序
├── streamlit_app.py          # Streamlit Web界面
├── requirements.txt          # 依赖列表
├── setup_env.sh             # 环境设置脚本
├── run_streamlit.sh         # Streamlit启动脚本
├── src/
│   ├── data/
│   │   └── data_loader.py   # 数据加载模块
│   ├── multimodal/
│   │   └── llava_processor.py  # LLaVA图片处理
│   ├── llm/
│   │   └── qwen_model.py    # Qwen模型集成
│   └── rag/
│       ├── vector_store.py   # 向量数据库
│       └── rag_system.py     # RAG系统
├── data/                    # 数据目录
│   ├── scienceqa/           # ScienceQA数据集
│   └── vectordb/           # 向量数据库存储
└── models/                  # 模型权重目录
```

## 注意事项

- 所有文件（包括模型权重）都存储在数据盘 `/root/autodl-tmp` 下
- 首次运行需要下载LLaVA模型（约13GB）和embedding模型
- Qwen API需要有效的DashScope API密钥

