# 快速开始指南

本指南将帮助您快速上手 ScienceQA RAG 项目。

## 前置要求

- Python 3.12 或更高版本
- CUDA 11.8+ (推荐使用 GPU)
- 至少 16GB 内存
- 至少 50GB 可用磁盘空间

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/BoPythonAI/Custom-Chatbots-with-LLMs-and-VLMs.git
cd Custom-Chatbots-with-LLMs-and-VLMs
```

### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env  # 如果存在示例文件
# 或手动创建
nano .env
```

添加以下内容：

```bash
DASHSCOPE_API_KEY=your_api_key_here
```

### 5. 准备数据

下载 ScienceQA 数据集并放置在 `data/scienceqa/` 目录：

- `problems.json`
- `captions.json`
- `pid_splits.json`
- `images/` 目录

### 6. 构建向量数据库

```bash
python main.py build_db
```

这个过程可能需要一些时间，取决于您的硬件配置。

### 7. 启动应用

**Web 界面：**
```bash
streamlit run streamlit_app.py
```

**命令行界面：**
```bash
python main.py interactive
```

## 下一步

- 查看 [完整项目报告](FINAL_PROJECT_REPORT.md) 了解项目详情
- 阅读 [训练指南](TRAINING_GUIDE.md) 学习如何微调模型
- 查看 [API 文档](API.md) 了解如何使用代码接口

## 常见问题

### Q: 如何获取 DashScope API Key?
A: 访问 [阿里云 DashScope](https://dashscope.aliyun.com/) 注册账号并获取 API Key。

### Q: 模型下载很慢怎么办?
A: 可以配置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: GPU 内存不足怎么办?
A: 可以在 `config.py` 中调整 `EMBEDDING_BATCH_SIZE` 和 `TRAINING_BATCH_SIZE`。

## 获取帮助

- 提交 [Issue](https://github.com/BoPythonAI/Custom-Chatbots-with-LLMs-and-VLMs/issues)
- 查看 [项目文档](FINAL_PROJECT_REPORT.md)

