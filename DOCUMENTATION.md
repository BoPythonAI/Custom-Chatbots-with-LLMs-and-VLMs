# ScienceQA RAG 多模态智能问答系统 - 完整项目文档

## 目录

1. [项目简介](#项目简介)
2. [快速开始](#快速开始)
3. [用户手册](#用户手册)
4. [API文档](#api文档)
5. [开发指南](#开发指南)
6. [配置说明](#配置说明)
7. [常见问题](#常见问题)
8. [故障排除](#故障排除)

---

## 项目简介

### 什么是 ScienceQA RAG？

ScienceQA RAG 是一个基于检索增强生成（Retrieval-Augmented Generation）技术的多模态智能问答系统。它能够：

- 📚 **智能检索**：从ScienceQA知识库中检索相关信息
- 🖼️ **图像理解**：使用LLaVA模型理解科学图像
- 💬 **智能回答**：基于检索到的上下文生成准确答案
- 🌐 **友好界面**：提供现代化的Web交互界面

### 核心特性

- ✅ **多模态支持**：同时支持文本和图像问题
- ✅ **检索增强**：基于向量数据库的语义检索
- ✅ **智能生成**：使用Qwen-max大语言模型生成答案
- ✅ **实时交互**：Streamlit Web界面，支持实时问答

### 技术栈

- **LLaVA-1.5-7B**: 多模态视觉-语言模型
- **Qwen-max**: 通义千问大语言模型
- **LangChain**: RAG框架
- **ChromaDB**: 向量数据库
- **Streamlit**: Web界面框架

---

## 快速开始

### 系统要求

- Python 3.8+
- CUDA支持的GPU（推荐，用于LLaVA模型）
- 至少16GB内存
- 约20GB磁盘空间（用于模型和数据）

### 安装步骤

#### 1. 克隆项目

```bash
cd /root/autodl-tmp/SQA
```

#### 2. 设置环境

```bash
chmod +x setup_env.sh
./setup_env.sh
```

这将：
- 创建Python虚拟环境
- 安装所有依赖包
- 配置环境变量

#### 3. 准备数据

确保ScienceQA数据集已放置在 `data/scienceqa/` 目录：

```
data/scienceqa/
├── problems.json      # 问题数据
├── captions.json     # 图片描述
├── pid_splits.json   # 数据集划分
└── images/          # 图片文件
```

#### 4. 构建向量数据库

```bash
source venv/bin/activate
python main.py build_db
```

#### 5. 启动Web界面

```bash
./run_streamlit.sh
```

访问 `http://localhost:8501`

---

## 用户手册

### Web界面使用

#### 界面布局

```
┌─────────────────────────────────────────────────┐
│  🔬 ScienceQA RAG 聊天机器人                    │
│  基于 LLaVA + LangChain + RAG                  │
├──────────────────────────┬──────────────────────┤
│                          │                      │
│   聊天消息区域            │   系统控制台         │
│                          │                      │
│   [用户消息]              │   ⚙️ 系统控制台     │
│   [助手回复]              │                      │
│                          │   🗑️ 清空聊天记录    │
│                          │                      │
│   ┌──────────────────┐   │   📊 系统状态        │
│   │ [上传图片] [输入框]│   │   ✅ RAG 知识库     │
│   │      [发送]      │   │   ✅ LLaVA 模型     │
│   └──────────────────┘   │   ✅ LangChain       │
│                          │                      │
│                          │   ⚙️ 模型配置        │
│                          │   📊 知识库统计      │
└──────────────────────────┴──────────────────────┘
```

#### 基本操作

**1. 文本问答**
- 在输入框中输入问题
- 点击"发送"按钮或按Enter键
- 系统会自动检索相关信息并生成答案

**2. 图像问答**
- 点击左侧的图片上传按钮
- 选择图片文件（支持PNG、JPG、JPEG）
- 输入相关问题（可选）
- 点击"发送"
- 系统会使用LLaVA分析图片并回答问题

**3. 配置调整**
- 在右侧控制台可以调整：
  - **检索文档数**：控制检索的相关文档数量（1-10）
  - **温度参数**：控制答案的创造性（0.0-1.0）
  - **科目过滤**：按科目筛选检索结果

**4. 清空聊天**
- 点击"🗑️ 清空聊天记录"按钮
- 聊天历史将被清空，系统会重新初始化

### 命令行使用

#### 交互式问答

```bash
python main.py interactive
```

进入交互模式后：
- 直接输入问题
- 输入 `quit` 或 `exit` 退出

#### 处理图片

```bash
# 处理所有图片
python main.py process_images

# 处理指定数量的图片（测试用）
python main.py process_images --max-images 10
```

#### 构建向量数据库

```bash
python main.py build_db
```

---

## API文档

### 核心类和方法

#### 1. ScienceQADataLoader

**位置**: `src/data/data_loader.py`

**功能**: 加载ScienceQA数据集

**主要方法**:

```python
loader = ScienceQADataLoader()

# 加载所有问题
problems = loader.load_problems()

# 加载图片描述
captions = loader.load_captions()

# 获取指定数据集划分
train_problems = loader.get_split_problems("train")
val_problems = loader.get_split_problems("val")
test_problems = loader.get_split_problems("test")
```

#### 2. LLaVAImageProcessor

**位置**: `src/multimodal/llava_processor.py`

**功能**: 处理图像并生成科学化描述

**主要方法**:

```python
processor = LLaVAImageProcessor()

# 生成图像描述
from PIL import Image
image = Image.open("path/to/image.jpg")
description = processor.generate_scientific_description(
    image, 
    question_context="题目: 这是什么？"
)

# 融合描述
merged = processor.merge_captions(
    official_caption="官方描述",
    caption_llava="LLaVA生成的描述"
)
```

**参数说明**:
- `image`: PIL Image对象
- `question_context`: 可选，题目上下文信息

#### 3. ScienceQAVectorStore

**位置**: `src/rag/vector_store.py`

**功能**: 管理向量数据库

**主要方法**:

```python
vector_store = ScienceQAVectorStore()

# 从问题数据生成文档
documents = vector_store.load_documents_from_problems(
    problems, captions, llava_captions
)

# 构建向量数据库
vector_store.build_vector_store(documents)

# 加载已存在的向量数据库
vector_store.load_vector_store()

# 相似度搜索
results = vector_store.similarity_search(
    query="什么是光合作用？",
    k=5,
    filter_dict={"subject": "biology"}  # 可选过滤
)
```

**参数说明**:
- `query`: 查询文本
- `k`: 返回的文档数量
- `filter_dict`: 元数据过滤条件

#### 4. QwenLLM

**位置**: `src/llm/qwen_model.py`

**功能**: 调用Qwen大语言模型

**主要方法**:

```python
llm = QwenLLM(api_key="your-api-key")

# 通用文本生成
response = llm.generate(
    prompt="请解释什么是重力",
    temperature=0.7,
    max_tokens=2048
)

# 回答ScienceQA问题
answer = llm.answer_question(
    question="什么是光合作用？",
    context="检索到的上下文信息",
    image_description="图片描述（可选）",
    choices=["选项A", "选项B", "选项C"]  # 可选
)
```

**参数说明**:
- `question`: 问题文本
- `context`: 检索到的上下文
- `image_description`: 图像描述（可选）
- `choices`: 选择题选项（可选）

#### 5. ScienceQARAGSystem

**位置**: `src/rag/rag_system.py`

**功能**: RAG系统核心，整合检索和生成

**主要方法**:

```python
rag_system = ScienceQARAGSystem(
    vector_store=vector_store,
    llm=llm,
    problems=problems,
    captions=captions,
    llava_captions=llava_captions
)

# 完整的RAG问答流程
result = rag_system.answer_with_rag(
    question="什么是光合作用？",
    choices=["选项A", "选项B"],  # 可选
    image_path="path/to/image.jpg",  # 可选
    problem_id="problem_123",  # 可选
    subject="biology",  # 可选
    k=5,  # 检索文档数
    auto_process_image=True  # 是否自动处理图片
)

# 返回结果包含：
# - answer: 生成的答案
# - retrieved_documents: 检索到的文档数量
# - context: 检索到的上下文
# - retrieved_docs: 检索到的文档列表
# - image_description: 图像描述（如果有）
# - is_image_question: 是否为图像问题
```

**参数说明**:
- `question`: 问题文本（必需）
- `choices`: 选择题选项列表（可选）
- `image_path`: 图片路径（可选）
- `problem_id`: 问题ID（可选）
- `subject`: 科目过滤（可选）
- `k`: 检索文档数量（可选，默认5）
- `auto_process_image`: 是否自动处理图片（默认True）

---

## 开发指南

### 项目结构

```
SQA/
├── config.py                 # 全局配置
├── main.py                   # 命令行入口
├── streamlit_app.py          # Web界面
├── requirements.txt          # 依赖列表
├── src/
│   ├── data/
│   │   └── data_loader.py   # 数据加载
│   ├── multimodal/
│   │   └── llava_processor.py  # LLaVA处理
│   ├── llm/
│   │   └── qwen_model.py    # Qwen集成
│   └── rag/
│       ├── vector_store.py   # 向量数据库
│       └── rag_system.py     # RAG系统
└── data/
    └── scienceqa/           # 数据集
```

### 代码规范

- 使用Python类型提示
- 遵循PEP 8代码风格
- 添加详细的docstring
- 模块化设计，职责单一

### 扩展开发

#### 添加新的Embedding模型

1. 在 `src/rag/embeddings/` 创建新的embedding类
2. 实现LangChain兼容的接口
3. 在 `config.py` 添加配置
4. 在 `vector_store.py` 中集成

#### 添加新的LLM模型

1. 在 `src/llm/` 创建新的LLM类
2. 实现 `generate()` 和 `answer_question()` 方法
3. 在 `config.py` 添加配置
4. 在 `rag_system.py` 中使用

#### 添加新的评估指标

1. 在 `src/evaluation/` 创建评估模块
2. 实现评估函数
3. 集成到评估流程中

---

## 配置说明

### 环境变量

创建 `.env` 文件（可选）：

```bash
# DashScope API密钥
DASHSCOPE_API_KEY=your-api-key-here

# HuggingFace镜像（可选）
HF_ENDPOINT=https://hf-mirror.com

# 模型配置（可选）
QWEN_MODEL=qwen-max
LLAVA_MODEL=llava-hf/llava-1.5-7b-hf
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# GPU配置（可选）
CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
```

### 配置文件 (`config.py`)

主要配置项：

```python
# API密钥
DASHSCOPE_API_KEY = "your-api-key"

# 模型配置
QWEN_MODEL = "qwen-max"
LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# RAG配置
TOP_K_RETRIEVAL = 5        # 检索文档数
TEMPERATURE = 0.7          # LLM温度参数
MAX_TOKENS = 2048          # 最大生成token数
CHUNK_SIZE = 1000          # 文本分割块大小
CHUNK_OVERLAP = 200        # 文本分割重叠大小

# GPU配置
NUM_GPUS = 2
CUDA_VISIBLE_DEVICES = "0,1"
```

---

## 常见问题

### Q1: 如何获取DashScope API密钥？

A: 
1. 访问阿里云DashScope控制台
2. 注册/登录账号
3. 创建API密钥
4. 在 `.env` 文件或 `config.py` 中配置

### Q2: LLaVA模型加载失败怎么办？

A: 
- 检查网络连接（需要访问HuggingFace）
- 确认有足够的磁盘空间（约13GB）
- 检查CUDA是否可用（如果使用GPU）
- 查看错误日志定位具体问题

### Q3: 向量数据库构建很慢？

A: 
- 这是正常现象，首次构建需要向量化所有文档
- 可以使用GPU加速embedding计算
- 构建完成后会持久化，后续直接加载即可

### Q4: Web界面无法访问？

A: 
- 检查端口8501是否被占用
- 确认防火墙设置
- 检查Streamlit服务是否正常启动
- 查看终端错误信息

### Q5: 如何提高答案质量？

A: 
- 增加检索文档数（k值）
- 调整温度参数（降低温度提高准确性）
- 使用科目过滤缩小检索范围
- 确保向量数据库已包含相关数据

### Q6: 支持哪些图片格式？

A: 
- PNG
- JPG/JPEG
- 其他PIL支持的格式

### Q7: 如何批量处理问题？

A: 
- 使用命令行交互模式
- 编写脚本调用API
- 使用Python代码直接调用RAG系统

---

## 故障排除

### 问题1: 模块导入错误

**症状**: `ImportError: No module named 'xxx'`

**解决方案**:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 问题2: CUDA内存不足

**症状**: `CUDA out of memory`

**解决方案**:
- 减少batch size
- 使用CPU模式
- 关闭其他占用GPU的程序
- 在 `config.py` 中设置使用CPU

### 问题3: API调用失败

**症状**: Qwen API返回错误

**解决方案**:
- 检查API密钥是否正确
- 确认账户余额充足
- 检查网络连接
- 查看API调用限制

### 问题4: 向量数据库损坏

**症状**: 无法加载向量数据库

**解决方案**:
```bash
# 删除旧的向量数据库
rm -rf data/vectordb/*

# 重新构建
python main.py build_db
```

### 问题5: Streamlit端口被占用

**症状**: `Port 8501 is already in use`

**解决方案**:
```bash
# 停止占用端口的进程
pkill -9 -f streamlit

# 或使用不同端口
streamlit run streamlit_app.py --server.port 8502
```

### 问题6: 图片处理失败

**症状**: LLaVA无法处理图片

**解决方案**:
- 检查图片文件是否存在
- 确认图片格式正确
- 检查LLaVA模型是否正常加载
- 查看错误日志

---

## 性能优化建议

### 1. 向量数据库优化

- 使用GPU加速embedding计算
- 优化chunk_size和chunk_overlap参数
- 定期重建向量数据库

### 2. 模型加载优化

- LLaVA模型延迟加载（已实现）
- 使用模型缓存避免重复下载
- GPU内存管理

### 3. 检索优化

- 使用元数据过滤减少搜索空间
- 调整检索文档数量
- 优化embedding模型选择

### 4. 生成优化

- 调整temperature和max_tokens参数
- 使用流式生成（如果支持）
- 缓存常见问题的答案

---

## 贡献指南

### 代码提交

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

### 代码审查

- 确保代码符合规范
- 添加必要的测试
- 更新文档

---

## 许可证

本项目遵循相应的开源许可证。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues]
- 邮箱: [联系邮箱]

---

**文档版本**: v1.0.0  
**最后更新**: 2025年11月24日

