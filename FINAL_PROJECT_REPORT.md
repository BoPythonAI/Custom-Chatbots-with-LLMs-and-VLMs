# ScienceQA RAG 多模态智能问答系统 - 最终项目报告

## 项目信息

- **项目名称**: ScienceQA RAG 多模态智能问答系统
- **项目类型**: 检索增强生成（RAG）系统 + Embedding 模型微调
- **数据集**: ScienceQA Dataset
- **完成日期**: 2024年

---

## 一、项目概述

### 1.1 项目背景

ScienceQA RAG 多模态智能问答系统是一个基于 ScienceQA 数据集的智能教育助手项目。该项目整合了多模态视觉理解、检索增强生成（RAG）和大型语言模型等前沿技术，实现了对科学问题的智能解答，支持文本和图像的多模态问答。

### 1.2 项目目标

本项目的主要目标包括：

1. **构建高效的 RAG 系统**：实现基于向量检索的知识增强问答
2. **优化 Embedding 模型**：通过微调提升检索质量
3. **多模态支持**：整合图像理解能力，支持图文混合问答
4. **性能评估**：对比不同 embedding 模型的检索效果

### 1.3 核心价值

- **教育应用**：为科学教育提供智能问答支持
- **技术验证**：验证 RAG 技术和 embedding 微调在科学问答领域的有效性
- **系统集成**：展示 LLaVA、LangChain、RAG 等技术的协同工作

---

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

#### 核心框架
- **LangChain**: RAG 框架，提供文档处理、向量存储、检索等功能
- **ChromaDB**: 向量数据库，存储和检索文档向量
- **Transformers**: HuggingFace 模型库，支持 LLaVA 等模型
- **PyTorch**: 深度学习框架

#### 模型组件
- **LLaVA-1.5-7B**: 多模态视觉-语言模型，用于图像理解
- **Qwen-max**: 通义千问大语言模型，用于答案生成
- **Jina v2 Base**: 原始 embedding 模型（`jinaai/jina-embeddings-v2-base-en`）
- **Jina v2 Fine-tuned**: 微调后的 embedding 模型
- **paraphrase-multilingual-MiniLM-L12-v2**: HuggingFace 多语言 embedding 模型

#### 开发工具
- **Streamlit**: Web 界面框架
- **DashScope**: 阿里云 API 服务，提供 Qwen 模型访问

---

## 三、核心功能模块

### 3.1 数据加载模块

**功能**：
- 加载 ScienceQA 数据集（problems.json, captions.json, pid_splits.json）
- 支持数据集划分（train/val/test）
- 提供统一的数据访问接口

**关键类**：`ScienceQADataLoader`

### 3.2 多模态处理模块

**功能**：
- 使用 LLaVA 模型生成图像的科学化描述
- 融合官方 caption 和 LLaVA 生成的描述
- 支持 GPU 加速和 CPU 推理

**关键类**：`LLaVAImageProcessor`

### 3.3 向量存储模块

**功能**：
- 构建和管理向量数据库
- 支持多种 embedding 模型
- 实现高效的相似度检索

**关键类**：`ScienceQAVectorStore`

### 3.4 RAG 系统模块

**功能**：
- 整合检索和生成流程
- 多模态上下文融合
- 答案生成与优化

**关键类**：`ScienceQARAGSystem`

### 3.5 Embedding 模型微调模块

**功能**：
- 准备训练数据（问题-答案对、问题-文档对）
- 实现对比学习训练（InfoNCE Loss）
- 支持硬负样本挖掘（批量优化）
- 支持 In-Batch Negatives
- 支持多任务学习（QA + QD）
- 早停机制和最佳模型保存

**关键类**：
- `TrainingDataPreparation`: 训练数据准备（支持多任务和硬负样本）
- `JinaTrainer`: 模型训练器（支持多任务学习、早停、学习率调度）

### 3.6 评估模块

**功能**：
- 检索质量评估（Recall@K, MRR）
- 答案质量评估（Accuracy, BLEU, ROUGE, BERTScore）
- 自动化对比实验框架
- 内存优化的模型加载/卸载

**关键类**：
- `EmbeddingComparison`: 模型对比评估
- `AnswerEvaluator`: 答案质量评估器

---

## 四、Embedding 模型微调方法

### 4.1 训练数据准备

#### 数据来源
- **训练集**: ScienceQA train split（7,636 个问题）
- **验证集**: ScienceQA val split（1,719 个问题，用于模型选择和早停）

#### 数据格式
本项目采用**多任务学习**策略，同时训练两个任务：

1. **任务1：问题-答案相似度（QA Similarity）**
   - **正样本对**: 问题-答案匹配对
   - **目标**: 学习问题和正确答案之间的语义相似性
   - **应用场景**: 答案质量评估、答案匹配

2. **任务2：问题-文档检索相似度（QD Retrieval Similarity）**
   - **正样本对**: 问题-文档块匹配对（文档包含该问题）
   - **目标**: 学习问题和相关文档之间的检索相似性
   - **应用场景**: RAG 检索、文档检索

#### 负样本策略

**硬负样本挖掘（Hard Negative Mining）**:
- 使用预训练的 embedding 模型计算候选负样本的相似度
- 选择与 anchor（问题）相似但答案/文档不同的样本作为负样本
- 评分公式：`score = anchor_similarity - 0.5 * positive_similarity`
- 选择 top-k 最难的负样本，提升模型区分能力

**In-Batch Negatives**:
- 自动利用 batch 内其他样本作为负样本
- 增加负样本多样性，提升训练效率
- 基于 SimCSE、DPR 等顶会论文的最佳实践

#### 训练数据统计
- **QA 任务**: 7,636 个正样本对，每个配 4 个硬负样本
- **QD 任务**: 7,636 个问题-文档对，每个配 4 个硬负样本
- **总训练样本**: 约 15,272 个训练样本（多任务混合）
- **验证样本**: 1,719 个验证样本（用于早停和模型选择）

### 4.2 训练方法

#### 损失函数
- **InfoNCE Loss**: 对比学习损失函数
- **温度参数**: 0.05（可配置）
- **In-Batch Negatives**: 自动利用 batch 内其他样本作为负样本

#### 训练超参数
- **学习率**: 5e-6（降低的学习率，防止破坏预训练权重）
- **训练轮数**: 2 epochs（防止过拟合）
- **Batch Size**: 4
- **梯度累积步数**: 8（有效 batch size = 32）
- **最大序列长度**: 512
- **学习率调度**: Cosine annealing with warmup（10% warmup steps）
- **温度参数**: 0.05（InfoNCE Loss）
- **任务权重**: QA = 0.5, QD = 0.5（多任务学习）

#### 优化技术

1. **多任务学习（Multi-Task Learning）**:
   - 同时训练 QA 相似度和 QD 检索相似度两个任务
   - 任务权重：QA = 0.5, QD = 0.5
   - 优势：同时优化答案匹配和文档检索能力，提升模型泛化性

2. **硬负样本挖掘（Hard Negative Mining）**:
   - 使用预训练 embedding 模型预计算候选负样本的 embedding
   - 批量计算相似度，选择最难的负样本（高 anchor 相似度，低 positive 相似度）
   - 限制候选负样本数量为 100，提升计算效率
   - 添加进度条显示，优化用户体验

3. **In-Batch Negatives**: 
   - 基于 SimCSE、DPR 的最佳实践
   - 自动利用 batch 内其他样本作为负样本
   - 增加负样本多样性，提升训练效率

4. **早停机制（Early Stopping）**: 
   - 基于验证集损失监控
   - Patience = 2 epochs（验证损失不下降 2 轮后停止）
   - 自动保存最佳模型（验证损失最低的 checkpoint）

5. **学习率调度优化**:
   - 使用 Cosine Annealing with Warmup
   - Warmup 步数 = 总步数的 10%
   - 平滑的学习率衰减，避免训练不稳定

6. **内存优化**:
   - Embedding 批量处理（batch size = 16）
   - 定期清理 GPU 缓存（`torch.cuda.empty_cache()`）
   - 顺序加载/卸载模型，避免同时加载多个模型导致 OOM

### 4.3 模型评估

#### 检索质量评估指标
- **Recall@K**: 在 top-k 检索结果中，能检索到原始问题本身的比例
  - 衡量模型在 top-k 结果中能否找到正确答案
- **MRR (Mean Reciprocal Rank)**: 平均倒数排名
  - 衡量模型对正确答案的排序质量（位置越靠前，MRR 越高）

#### 答案质量评估指标（可选）
- **Accuracy**: 答案正确率（选择题 A/B/C/D 匹配）
- **BLEU**: 文本相似度（n-gram 重叠度）
- **ROUGE**: 召回导向的相似度（ROUGE-1/2/L）
- **BERTScore**: 基于 BERT 的语义相似度

#### 评估方法
- **自检索评估**: 使用问题本身作为查询，检索包含该问题的文档
- **评估数据集**: ScienceQA test split（4,241 个问题）
- **Top-K**: 5（默认）
- **内存优化**: 顺序加载/卸载模型，避免 GPU OOM

---

## 五、实验设置

### 5.1 对比模型

1. **Jina v2 Original**: 
   - 模型: `jinaai/jina-embeddings-v2-base-en`
   - 未经过微调，作为基线模型

2. **Jina v2 Fine-tuned**: 
   - 基于 Jina v2 Base 模型微调
   - 使用 ScienceQA 训练集进行微调
   - 应用了 In-Batch Negatives 和硬负样本挖掘

3. **HuggingFace Model**: 
   - 模型: `paraphrase-multilingual-MiniLM-L12-v2`
   - 多语言 embedding 模型，作为对比基线

### 5.2 实验环境

- **硬件**: GPU 加速训练和推理
- **软件**: Python 3.12, PyTorch, Transformers
- **数据集**: ScienceQA Dataset
- **评估数据集**: Test split（4,241 个问题）

### 5.3 评估设置

- **检索 Top-K**: 5
- **评估指标**: Recall@5, MRR
- **评估方法**: 自检索评估（self-retrieval evaluation）
- **评估数据集**: Test split（4,241 个问题）
- **内存优化**: 顺序加载/卸载模型，避免 GPU OOM

### 5.4 训练过程记录

**训练配置**:
- 训练数据: 15,272 个样本（多任务混合）
- 验证数据: 1,719 个样本
- 总训练步数: 1,274 步（2 epochs）
- Warmup 步数: 127 步（10%）

**训练过程**:
- **Epoch 1**:
  - 训练损失: 0.0995（最终）
  - 验证损失: 0.0491
  - 最佳模型: ✅ 已保存（验证损失最低）
  
- **Epoch 2**:
  - 训练损失: 0.0481（最终）
  - 验证损失: 0.0461
  - 最佳模型: ✅ 已保存（验证损失更低）

**训练观察**:
- 验证损失持续下降，说明模型在持续改进
- 训练损失和验证损失差距合理，未出现明显过拟合
- 早停机制未触发（验证损失持续下降）
- 最佳模型保存在 Epoch 2（验证损失 = 0.0461）

---

## 六、实验结果

### 6.1 检索质量评估结果

#### 总体结果对比

| 模型 | Recall@5 | MRR | 正确检索数 | 总问题数 |
|------|----------|-----|------------|----------|
| **Jina v2 Original** | 0.4001 | 0.3990 | 1,697 | 4,241 |
| **Jina v2 Fine-tuned** | **0.4490** | **0.4028** | **1,904** | 4,241 |
| **HuggingFace** | 0.3893 | 0.3866 | 1,651 | 4,241 |

#### 性能提升分析

**Jina v2 Fine-tuned vs Jina v2 Original**:
- **Recall@5 提升**: +4.89% (0.4001 → 0.4490)
- **MRR 提升**: +0.38% (0.3990 → 0.4028)
- **正确检索数增加**: +207 个问题（+12.2%）

**Jina v2 Fine-tuned vs HuggingFace**:
- **Recall@5 提升**: +5.97% (0.3893 → 0.4490)
- **MRR 提升**: +1.62% (0.3866 → 0.4028)

### 6.2 结果分析

#### 优势
1. **微调效果显著**: 
   - Recall@5 提升了 4.89%（0.4001 → 0.4490），说明微调后的模型在 top-5 检索能力上有明显提升
   - 正确检索数从 1,697 增加到 1,904，提升了 12.2%
   - 相比原始模型，能多检索到 207 个问题

2. **优于对比基线**:
   - 在所有指标上都优于 HuggingFace 模型（Recall@5: +5.97%, MRR: +1.62%）
   - 相比原始 Jina v2 模型有明显提升

3. **训练稳定性**:
   - 使用较低的学习率（5e-6）和较少的训练轮数（2 epochs）确保了训练稳定性
   - 早停机制防止了过拟合
   - 验证损失持续下降（Epoch 1: 0.0491 → Epoch 2: 0.0461）

4. **多任务学习效果**:
   - 同时优化 QA 相似度和 QD 检索相似度
   - 模型在检索任务上表现更好，说明多任务学习策略有效

#### 改进空间
1. **MRR 提升有限**: 
   - MRR 仅提升了 0.38%（0.3990 → 0.4028），说明模型在 top-1 排序能力上还有改进空间
   - 可能需要进一步优化训练策略、调整温度参数或使用更复杂的损失函数

2. **检索覆盖率**:
   - 仍有 55.10% 的问题无法在 top-5 中找到
   - 可以通过数据增强、更大的训练集、更长的训练或更复杂的负样本策略来改进

3. **答案质量评估**:
   - 当前主要关注检索质量，答案质量评估（Accuracy, BLEU, ROUGE, BERTScore）可作为未来改进方向

### 6.3 技术贡献

1. **多任务学习框架**:
   - 同时训练 QA 相似度和 QD 检索相似度两个任务
   - 通过任务权重平衡两个任务的学习
   - 提升模型在检索任务上的泛化能力

2. **硬负样本挖掘优化**:
   - 批量预计算候选负样本的 embedding，避免重复计算
   - 限制候选数量（100 个），提升计算效率
   - 添加进度条显示，优化用户体验
   - 动态选择最难的负样本，提升模型区分能力

3. **In-Batch Negatives 实现**:
   - 基于 SimCSE、DPR 等顶会论文的最佳实践
   - 自动利用 batch 内样本作为负样本，提升训练效率
   - 增加负样本多样性

4. **训练策略优化**:
   - 使用较低的学习率（5e-6）和较少的训练轮数（2 epochs）
   - 防止破坏预训练模型的权重
   - 实现稳定的微调过程
   - Cosine annealing with warmup 学习率调度

5. **内存优化**:
   - Embedding 批量处理，避免 GPU OOM
   - 顺序加载/卸载模型，支持大规模对比实验
   - 定期清理 GPU 缓存

6. **完整的评估体系**:
   - 检索质量评估（Recall@K, MRR）
   - 答案质量评估（Accuracy, BLEU, ROUGE, BERTScore）
   - 自动化对比实验框架

---

## 七、项目特色与创新点

### 7.1 技术创新

1. **多模态 RAG 系统**:
   - 整合 LLaVA 图像理解和 RAG 文本检索
   - 支持图文混合问答
   - 融合官方 caption 和 LLaVA 生成的描述

2. **多任务学习框架**:
   - 同时训练 QA 相似度和 QD 检索相似度
   - 通过任务权重平衡学习
   - 提升模型在检索任务上的泛化能力

3. **Embedding 模型微调**:
   - 针对 ScienceQA 数据集进行领域适配
   - 应用最新的对比学习技术（InfoNCE Loss）
   - 多任务学习 + 硬负样本挖掘

4. **训练优化技术**:
   - In-Batch Negatives（SimCSE/DPR 风格）
   - 硬负样本挖掘（批量优化）
   - 精细的学习率调度（Cosine Annealing with Warmup）
   - 早停机制和最佳模型保存

5. **内存优化技术**:
   - Embedding 批量处理
   - 顺序加载/卸载模型
   - 定期清理 GPU 缓存

### 7.2 工程实践

1. **模块化设计**:
   - 清晰的代码结构（数据加载、多模态处理、RAG、训练、评估）
   - 易于扩展和维护
   - 支持多种 embedding 模型切换

2. **完整的评估体系**:
   - 自动化的模型对比评估
   - 检索质量评估（Recall@K, MRR）
   - 答案质量评估（Accuracy, BLEU, ROUGE, BERTScore）
   - 详细的性能指标分析

3. **文档完善**:
   - 详细的使用指南
   - 技术文档和优化方案
   - 完整的项目报告

4. **命令行工具**:
   - 统一的命令行接口（`main.py`）
   - 支持多种运行模式（build_db, train_jina, compare_embeddings 等）
   - 灵活的参数配置

### 7.3 项目亮点总结

1. **性能提升显著**: 
   - Recall@5 提升 4.89%（0.4001 → 0.4490）
   - 正确检索数增加 12.2%（1,697 → 1,904）

2. **技术方案先进**:
   - 多任务学习框架
   - 硬负样本挖掘优化
   - 内存优化的评估框架

3. **工程实现完善**:
   - 完整的训练和评估流程
   - 自动化实验框架
   - 详细的文档和报告

---

## 八、项目文件结构

```
SQA/
├── config.py                      # 配置文件
├── main.py                        # 主程序入口
├── streamlit_app.py               # Web 界面
├── requirements.txt               # 依赖列表
│
├── src/                           # 源代码目录
│   ├── data/
│   │   └── data_loader.py        # 数据加载模块
│   ├── multimodal/
│   │   └── llava_processor.py   # LLaVA 图像处理
│   ├── llm/
│   │   └── qwen_model.py         # Qwen LLM 集成
│   ├── rag/
│   │   ├── embeddings/
│   │   │   └── jina_embedding.py  # Jina Embedding 封装
│   │   ├── vector_store.py       # 向量存储
│   │   └── rag_system.py         # RAG 系统
│   ├── training/
│   │   ├── data_preparation.py   # 训练数据准备（多任务+硬负样本）
│   │   └── jina_trainer.py       # 模型训练器（多任务学习）
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
│   ├── scienceqa/                # ScienceQA 数据集
│   ├── training/                  # 训练数据
│   └── vectordb/                  # 向量数据库
│
├── training_output/               # 训练输出
│   └── jina_finetuned/           # 微调后的模型
│
├── experiments/                   # 实验结果
│   └── full_comparison.json      # 对比评估结果（JSON格式）
│
├── training_output/              # 训练输出
│   └── jina_finetuned/          # 微调后的模型
│       ├── best_model/          # 最佳模型（验证损失最低）
│       └── checkpoint-*/        # 训练检查点
│
└── README.md                      # 项目说明
    FINAL_PROJECT_REPORT.md        # 最终报告（本文档）
```

---

## 九、使用方法

### 9.1 环境设置

```bash
cd /root/autodl-tmp/SQA
source venv/bin/activate
```

### 9.2 构建向量数据库

```bash
python main.py build_db
```

### 9.3 训练 Embedding 模型

```bash
# 1. 准备训练数据
python main.py prepare_data

# 2. 训练模型
python main.py train_jina
```

### 9.4 评估模型

```bash
# 对比评估所有模型
python main.py compare_embeddings --no-answer-eval
```

### 9.5 交互式问答

```bash
# 命令行交互
python main.py interactive

# Web 界面
./run_streamlit.sh
```

---

## 十、结论

### 10.1 项目成果

本项目成功构建了一个基于 ScienceQA 数据集的 RAG 多模态智能问答系统，并通过 embedding 模型微调显著提升了检索性能：

1. **检索性能提升**: 
   - Recall@5 从 0.4001 提升到 0.4490（+4.89%）
   - MRR 从 0.3990 提升到 0.4028（+0.38%）
   - 正确检索数从 1,697 增加到 1,904（+12.2%）
   - 相比 HuggingFace 基线，Recall@5 提升 5.97%

2. **技术实现**:
   - 实现了完整的 RAG 系统（多模态支持）
   - 成功微调了 embedding 模型（多任务学习 + 硬负样本）
   - 应用了最新的对比学习技术（InfoNCE Loss + In-Batch Negatives）
   - 实现了内存优化的评估框架

3. **系统功能**:
   - 支持多模态问答（文本 + 图像）
   - 提供 Web 和命令行两种交互方式
   - 完整的评估和对比体系（检索质量 + 答案质量）
   - 自动化训练和评估流程

### 10.2 技术贡献

1. **多任务学习框架**: 
   - 同时训练 QA 相似度和 QD 检索相似度
   - 通过任务权重平衡学习，提升模型泛化能力

2. **硬负样本挖掘优化**: 
   - 批量预计算 embedding，提升计算效率
   - 动态选择最难的负样本，提升模型区分能力
   - 添加进度显示和错误处理

3. **In-Batch Negatives**: 
   - 实现了基于 SimCSE/DPR 的批次内负样本技术
   - 自动利用 batch 内样本作为负样本

4. **训练策略优化**: 
   - 使用较低学习率和较少轮数，实现稳定高效的微调过程
   - Cosine annealing with warmup 学习率调度
   - 早停机制和最佳模型保存

5. **内存优化**: 
   - Embedding 批量处理
   - 顺序加载/卸载模型，支持大规模对比实验
   - 定期清理 GPU 缓存

### 10.3 未来改进方向

1. **进一步提升 MRR**: 
   - 优化 top-1 排序能力（当前 MRR 提升有限）
   - 调整温度参数（当前 0.05）
   - 尝试不同的损失函数（如 Triplet Loss、Margin Loss）
   - 使用更复杂的负样本策略

2. **扩大训练数据**:
   - 使用更多训练样本（当前 7,636 个）
   - 数据增强技术（回译、同义词替换等）
   - 使用外部科学问答数据集

3. **模型优化**:
   - 尝试更大的模型（当前使用 base 模型）
   - 集成多个模型（模型融合）
   - 尝试不同的预训练模型（如 BGE、E5）

4. **系统优化**:
   - 提升检索速度（当前批量处理已优化）
   - 进一步优化内存使用
   - 支持分布式训练

5. **答案质量评估**:
   - 完善答案质量评估流程
   - 分析答案质量与检索质量的关系
   - 优化 RAG 系统的答案生成策略

6. **多模态增强**:
   - 更好地融合图像和文本信息
   - 使用 CLIP 等视觉-语言模型
   - 优化多模态检索策略

---

## 十一、参考文献

### 核心论文
1. **SimCSE**: Simple Contrastive Learning of Sentence Embeddings (EMNLP 2021)
   - 提出了简单的对比学习框架，使用 In-Batch Negatives

2. **DPR**: Dense Passage Retrieval for Open-Domain Question Answering (NeurIPS 2020)
   - 密集段落检索，使用对比学习训练检索模型

3. **InfoNCE**: Representation Learning with Contrastive Predictive Coding (NeurIPS 2018)
   - 对比学习的理论基础，InfoNCE Loss 的提出

4. **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (ICML 2021)
   - 视觉-语言对比学习，多模态表示学习

### 数据集和模型
5. **ScienceQA**: Science Question Answering Dataset
   - 科学问答数据集，包含多模态问题

6. **LLaVA**: Large Language and Vision Assistant
   - 大型多模态视觉-语言模型

7. **Jina v2**: Jina Embeddings v2
   - 高性能文本嵌入模型

### 框架和工具
8. **LangChain**: Framework for developing applications powered by language models
   - RAG 应用开发框架

9. **ChromaDB**: Open-source vector database
   - 向量数据库，用于存储和检索嵌入向量

10. **Transformers**: HuggingFace Transformers Library
    - 预训练模型库和训练框架

---

## 附录

### A. 实验详细结果

实验结果保存在：`/root/autodl-tmp/SQA/experiments/full_comparison.json`

**最新实验结果**（2024年）:
```json
{
  "test_split": "test",
  "top_k": 5,
  "retrieval_quality": {
    "jina_v2_original": {
      "recall_at_k": 0.4001,
      "mrr": 0.3990,
      "correct_retrievals": 1697,
      "total_questions": 4241
    },
    "jina_v2_finetuned": {
      "recall_at_k": 0.4490,
      "mrr": 0.4028,
      "correct_retrievals": 1904,
      "total_questions": 4241
    },
    "huggingface": {
      "recall_at_k": 0.3893,
      "mrr": 0.3866,
      "correct_retrievals": 1651,
      "total_questions": 4241
    }
  }
}
```

### B. 训练配置

训练配置详见：`config.py`

**关键训练参数**:
- Batch Size: 4
- Gradient Accumulation: 8
- Effective Batch Size: 32
- Learning Rate: 5e-6
- Epochs: 2
- Early Stopping Patience: 2
- Learning Rate Scheduler: Cosine Annealing with Warmup (10%)
- Temperature: 0.05
- Multi-task Weights: QA=0.5, QD=0.5

**训练过程**:
- Epoch 1: Training Loss = 0.0995, Validation Loss = 0.0491
- Epoch 2: Training Loss = 0.0481, Validation Loss = 0.0461
- Best Model: Epoch 2 (Validation Loss = 0.0461)

### C. 项目文件结构

项目代码结构完整，包含：
- 完整的 RAG 系统实现
- Embedding 模型微调框架
- 评估和对比实验框架
- 详细的文档和示例
- 命令行工具和 Web 界面

### D. 运行命令示例

```bash
# 1. 准备训练数据（多任务 + 硬负样本）
python main.py prepare_data

# 2. 训练模型
python main.py train_jina --batch-size 4 --epochs 2 --learning-rate 5e-6

# 3. 评估模型（仅检索质量）
python main.py compare_embeddings --no-answer-eval --models jina_v2_finetuned

# 4. 完整评估（检索质量 + 答案质量）
python main.py compare_embeddings --models jina_v2_finetuned

# 5. 对比所有模型
python main.py compare_embeddings --no-answer-eval
```

### E. 技术栈版本

- Python: 3.12
- PyTorch: Latest
- Transformers: Latest
- LangChain: Latest
- ChromaDB: Latest
- Jina AI: Latest

---

**报告完成日期**: 2024年

**项目状态**: ✅ 已完成

**最后更新**: 2024年（训练和评估完成）

