# 项目总结

## 📋 项目信息

- **项目名称**: ScienceQA RAG 多模态智能问答系统
- **GitHub**: https://github.com/BoPythonAI/Custom-Chatbots-with-LLMs-and-VLMs
- **许可证**: MIT License
- **Python 版本**: 3.12+

## 🎯 核心功能

1. **多模态 RAG 系统**
   - LLaVA 图像理解
   - 向量检索
   - Qwen-max 答案生成

2. **Embedding 模型微调**
   - 多任务学习（QA + QD）
   - 硬负样本挖掘
   - 性能提升 4.89%

3. **完整评估体系**
   - 检索质量评估
   - 答案质量评估

## 📊 实验结果

| 模型 | Recall@5 | MRR |
|------|----------|-----|
| Jina v2 Original | 0.4001 | 0.3990 |
| **Jina v2 Fine-tuned** | **0.4490** | **0.4028** |

## 📁 项目结构

```
├── README.md              # 项目说明
├── LICENSE                # MIT 许可证
├── CONTRIBUTING.md        # 贡献指南
├── FINAL_PROJECT_REPORT.md  # 完整报告
├── requirements.txt       # 依赖列表
├── config.py             # 配置文件
├── main.py               # 主程序
├── src/                  # 源代码
├── scripts/              # 脚本
├── docs/                 # 文档
└── .github/              # GitHub Actions
```

## 🚀 快速开始

```bash
# 1. 克隆项目
git clone https://github.com/BoPythonAI/Custom-Chatbots-with-LLMs-and-VLMs.git

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
echo "DASHSCOPE_API_KEY=your_key" > .env

# 4. 构建向量数据库
python main.py build_db

# 5. 启动应用
streamlit run streamlit_app.py
```

## 📚 文档

- [README.md](README.md) - 项目说明
- [FINAL_PROJECT_REPORT.md](FINAL_PROJECT_REPORT.md) - 完整项目报告
- [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) - 快速开始指南
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)
