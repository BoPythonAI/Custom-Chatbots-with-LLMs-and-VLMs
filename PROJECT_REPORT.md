# ScienceQA RAG 多模态智能问答系统项目报告

## 一、项目概述

### 1.1 项目简介

ScienceQA RAG（Retrieval-Augmented Generation）多模态智能问答系统是一个基于ScienceQA数据集的智能教育助手项目。该项目整合了多模态视觉理解、检索增强生成和大型语言模型等前沿技术，实现了对科学问题的智能解答，支持文本和图像的多模态问答。

### 1.2 项目目标

- **多模态理解**：通过LLaVA模型实现图像的科学化描述生成
- **知识检索**：利用向量数据库实现高效的知识检索
- **智能生成**：基于检索到的上下文，使用Qwen LLM生成准确答案
- **用户交互**：提供友好的Web界面和命令行交互方式

### 1.3 核心价值

1. **教育应用**：为科学教育提供智能问答支持
2. **技术验证**：验证多模态RAG在科学问答领域的有效性
3. **系统集成**：展示LLaVA、LangChain、RAG等技术的协同工作

## 二、技术架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户交互层                              │
│  ┌──────────────┐          ┌──────────────┐            │
│  │ Streamlit Web│          │ 命令行交互   │            │
│  │    界面      │          │    (CLI)     │            │
│  └──────────────┘          └──────────────┘            │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  RAG系统核心层                           │
│  ┌──────────────────────────────────────────────────┐  │
│  │         ScienceQARAGSystem                        │  │
│  │  - 问题理解与处理                                 │  │
│  │  - 多模态上下文融合                               │  │
│  │  - 答案生成与优化                                 │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         ↓                    ↓                    ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  向量检索    │  │  多模态处理   │  │  LLM生成     │
│  VectorStore │  │  LLaVA       │  │  Qwen-max    │
└──────────────┘  └──────────────┘  └──────────────┘
         ↓                    ↓
┌──────────────┐  ┌──────────────┐
│  ChromaDB    │  │  Embedding   │
│  向量数据库  │  │  模型        │
└──────────────┘  └──────────────┘
```

### 2.2 技术栈

#### 2.2.1 核心框架
- **LangChain**: RAG框架，提供文档处理、向量存储、检索等功能
- **ChromaDB**: 向量数据库，存储和检索文档向量
- **Transformers**: HuggingFace模型库，支持LLaVA等模型

#### 2.2.2 模型组件
- **LLaVA-1.5-7B**: 多模态视觉-语言模型，用于图像理解
- **Qwen-max**: 通义千问大语言模型，用于答案生成
- **paraphrase-multilingual-MiniLM-L12-v2**: 多语言embedding模型

#### 2.2.3 开发工具
- **Streamlit**: Web界面框架
- **DashScope**: 阿里云API服务，提供Qwen模型访问
- **PyTorch**: 深度学习框架

## 三、核心功能模块

### 3.1 数据加载模块 (`src/data/data_loader.py`)

**功能**：
- 加载ScienceQA数据集（problems.json, captions.json, pid_splits.json）
- 支持数据集划分（train/val/test）
- 提供统一的数据访问接口

**关键类**：
- `ScienceQADataLoader`: 数据加载器类

**主要方法**：
- `load_problems()`: 加载问题数据
- `load_captions()`: 加载图片描述数据
- `get_split_problems()`: 获取指定数据集划分

### 3.2 多模态处理模块 (`src/multimodal/llava_processor.py`)

**功能**：
- 使用LLaVA模型生成图像的科学化描述
- 融合官方caption和LLaVA生成的描述
- 支持GPU加速和CPU推理

**关键类**：
- `LLaVAImageProcessor`: LLaVA图片处理器

**主要方法**：
- `generate_scientific_description()`: 生成科学化图像描述
- `merge_captions()`: 融合多种图像描述

**技术特点**：
- 使用`LlavaForConditionalGeneration`模型
- 支持对话式图像理解
- 自动设备检测（CUDA/CPU）

### 3.3 向量数据库模块 (`src/rag/vector_store.py`)

**功能**：
- 构建和管理ScienceQA向量数据库
- 实现相似度检索
- 支持元数据过滤（科目、主题等）

**关键类**：
- `ScienceQAVectorStore`: 向量数据库管理类

**主要方法**：
- `load_documents_from_problems()`: 从问题数据生成文档
- `build_vector_store()`: 构建向量数据库
- `similarity_search()`: 相似度搜索

**技术特点**：
- 使用HuggingFaceEmbeddings进行文本向量化
- 使用RecursiveCharacterTextSplitter进行文本分割
- 支持ChromaDB持久化存储

### 3.4 LLM模块 (`src/llm/qwen_model.py`)

**功能**：
- 集成Qwen-max大语言模型
- 生成基于上下文的答案
- 支持多模态prompt构建

**关键类**：
- `QwenLLM`: Qwen模型封装类

**主要方法**：
- `generate()`: 通用文本生成
- `answer_question()`: 回答ScienceQA问题

**技术特点**：
- 通过DashScope API调用Qwen模型
- 支持温度参数、最大token数等配置
- 智能prompt构建，融合上下文和图像描述

### 3.5 RAG系统模块 (`src/rag/rag_system.py`)

**功能**：
- 整合向量检索和LLM生成
- 实现多模态RAG流程
- 支持图像问题的自动处理

**关键类**：
- `ScienceQARAGSystem`: RAG系统核心类

**主要方法**：
- `retrieve_context()`: 检索相关上下文
- `format_context()`: 格式化检索结果
- `answer_with_rag()`: 完整的RAG问答流程

**工作流程**：
1. 问题理解（识别图像问题、提取问题信息）
2. 图像处理（调用LLaVA生成描述）
3. 上下文检索（向量相似度搜索）
4. 答案生成（LLM基于上下文生成答案）

### 3.6 Web界面模块 (`streamlit_app.py`)

**功能**：
- 提供现代化的Web交互界面
- 支持文本和图像输入
- 实时显示系统状态和配置

**界面特性**：
- 深色主题设计
- 文件上传组件嵌入输入框
- 聊天式对话界面
- 系统控制台（状态监控、参数配置）

## 四、项目结构

```
SQA/
├── config.py                 # 全局配置文件
├── main.py                   # 主程序入口
├── streamlit_app.py          # Streamlit Web界面
├── requirements.txt          # Python依赖列表
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
│   │   ├── problems.json    # 问题数据
│   │   ├── captions.json   # 图片描述
│   │   ├── pid_splits.json # 数据集划分
│   │   └── images/         # 图片文件
│   └── vectordb/           # 向量数据库存储
└── models/                  # 模型权重目录
```

## 五、工作流程

### 5.1 系统初始化流程

```
1. 加载ScienceQA数据集
   ├── problems.json (问题数据)
   ├── captions.json (图片描述)
   └── pid_splits.json (数据集划分)

2. 构建向量数据库
   ├── 文档生成（问题+选项+解释+描述）
   ├── 文本分割（chunk_size=1000, overlap=200）
   ├── 向量化（使用embedding模型）
   └── 存储到ChromaDB

3. 加载模型
   ├── LLaVA模型（延迟加载）
   ├── Embedding模型
   └── Qwen API连接
```

### 5.2 问答流程

```
用户输入问题
    ↓
判断是否为图像问题
    ├── 是 → 调用LLaVA生成图像描述
    └── 否 → 直接使用文本问题
    ↓
构建查询（问题+选项）
    ↓
向量检索（相似度搜索）
    ├── 支持科目过滤
    ├── 返回Top-K相关文档
    └── 格式化上下文
    ↓
构建Prompt
    ├── 系统提示
    ├── 检索到的上下文
    ├── 图像描述（如果有）
    ├── 问题文本
    └── 选项（如果有）
    ↓
LLM生成答案
    ↓
返回结果
    ├── 答案文本
    ├── 检索文档数量
    ├── 上下文信息
    └── 图像信息（如果有）
```

## 六、技术亮点

### 6.1 多模态融合

- **图像理解**：使用LLaVA模型生成科学化的图像描述
- **描述融合**：融合官方caption和LLaVA描述，提供更丰富的图像信息
- **上下文整合**：将图像描述与检索到的文本上下文结合，生成综合答案

### 6.2 检索增强生成

- **向量检索**：使用embedding模型将问题和文档向量化，实现语义相似度检索
- **元数据过滤**：支持按科目、主题等元数据进行检索过滤
- **上下文优化**：格式化检索结果，为LLM提供结构化上下文

### 6.3 系统设计

- **模块化架构**：清晰的模块划分，便于维护和扩展
- **延迟加载**：LLaVA模型采用延迟加载，减少启动时间
- **错误处理**：完善的异常处理机制，保证系统稳定性

### 6.4 用户体验

- **现代化界面**：深色主题、流畅动画、响应式设计
- **多输入方式**：支持文本输入和图像上传
- **实时反馈**：显示系统状态、检索信息、处理进度

## 七、配置说明

### 7.1 模型配置

- **QWEN_MODEL**: `qwen-max` - Qwen大语言模型
- **LLAVA_MODEL**: `llava-hf/llava-1.5-7b-hf` - LLaVA多模态模型
- **EMBEDDING_MODEL**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` - 多语言embedding模型

### 7.2 RAG配置

- **TOP_K_RETRIEVAL**: 5 - 检索文档数量
- **TEMPERATURE**: 0.7 - LLM温度参数
- **MAX_TOKENS**: 2048 - 最大生成token数
- **CHUNK_SIZE**: 1000 - 文本分割块大小
- **CHUNK_OVERLAP**: 200 - 文本分割重叠大小

### 7.3 硬件配置

- **NUM_GPUS**: 2 - GPU数量
- **CUDA_VISIBLE_DEVICES**: "0,1" - 可见GPU设备

## 八、使用说明

### 8.1 环境准备

```bash
# 1. 克隆项目（如果适用）
cd /root/autodl-tmp/SQA

# 2. 设置环境
chmod +x setup_env.sh
./setup_env.sh

# 3. 激活虚拟环境
source venv/bin/activate
```

### 8.2 数据准备

确保ScienceQA数据集已放置在 `data/scienceqa/` 目录下，包含：
- `problems.json`
- `captions.json`
- `pid_splits.json`
- `images/` 目录

### 8.3 构建向量数据库

```bash
python main.py build_db
```

### 8.4 处理图片（可选）

```bash
# 处理所有图片
python main.py process_images

# 处理指定数量图片（测试用）
python main.py process_images --max-images 10
```

### 8.5 启动Web界面

```bash
./run_streamlit.sh
```

访问 `http://localhost:8501`

### 8.6 命令行交互

```bash
python main.py interactive
```

## 九、项目特色

### 9.1 多模态支持

- 同时支持纯文本问题和图像问题
- 自动识别图像问题并调用LLaVA处理
- 融合多种图像描述来源

### 9.2 知识检索

- 基于向量相似度的语义检索
- 支持元数据过滤（科目、主题等）
- 可配置的检索文档数量

### 9.3 智能生成

- 基于检索上下文的答案生成
- 支持选择题和开放性问题
- 可配置的生成参数

### 9.4 用户友好

- 现代化的Web界面
- 实时系统状态显示
- 灵活的配置选项

## 十、技术挑战与解决方案

### 10.1 挑战：多模态数据融合

**解决方案**：
- 使用LLaVA生成图像描述
- 将图像描述作为文本上下文的一部分
- 在prompt中明确标注图像描述来源

### 10.2 挑战：检索质量

**解决方案**：
- 使用多语言embedding模型支持中英文
- 文本分割策略优化（chunk_size和overlap）
- 支持元数据过滤提高检索精度

### 10.3 挑战：系统性能

**解决方案**：
- LLaVA模型延迟加载
- GPU加速支持
- 向量数据库持久化存储

## 十一、项目总结

### 11.1 技术成果

1. **成功集成**：LLaVA、LangChain、ChromaDB、Qwen等技术的完整集成
2. **多模态RAG**：实现了图像-文本联合的检索增强生成
3. **系统化设计**：模块化、可扩展的系统架构

### 11.2 应用价值

1. **教育领域**：为科学教育提供智能问答支持
2. **技术验证**：验证多模态RAG在特定领域的有效性
3. **研究基础**：为后续研究和改进提供基础框架

### 11.3 未来改进方向

1. **模型优化**：fine-tuning embedding模型提升检索质量
2. **多模态增强**：实现图像-文本联合检索
3. **评估体系**：建立完整的评估指标和对比实验框架
4. **性能优化**：优化检索速度和生成效率

## 十二、依赖清单

### 12.1 核心依赖

- `torch>=2.0.0` - PyTorch深度学习框架
- `transformers>=4.35.0` - HuggingFace模型库
- `langchain>=0.2.0` - LangChain RAG框架
- `langchain-community>=0.2.0` - LangChain社区集成
- `langchain-core>=0.2.0` - LangChain核心库

### 12.2 向量数据库

- `chromadb>=0.4.0` - ChromaDB向量数据库
- `sentence-transformers>=2.2.0` - Sentence Transformers

### 12.3 多模态支持

- `Pillow>=10.0.0` - 图像处理
- `opencv-python>=4.8.0` - 计算机视觉库

### 12.4 API服务

- `dashscope>=1.17.0` - 阿里云DashScope API

### 12.5 Web界面

- `streamlit>=1.28.0` - Streamlit Web框架

## 十三、代码统计

- **总文件数**：约10个核心Python文件
- **代码行数**：约1500+行
- **模块数**：5个核心模块
- **类数**：6个主要类

## 十四、项目状态

- **开发状态**：✅ 核心功能已完成
- **测试状态**：✅ 基本功能已验证
- **文档状态**：✅ README和代码注释完善
- **部署状态**：✅ 支持本地部署和运行

---

**报告生成时间**：2025年11月24日  
**项目版本**：v1.0.0  
**维护者**：ScienceQA RAG Team

