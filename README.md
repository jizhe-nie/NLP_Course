# NLP_Course
# 📧 基于多架构演进的中文垃圾邮件拦截系统 (Spam Classification)

本项目是一个完整的自然语言处理（NLP）横向对比工程。旨在通过对 10000 条中文邮件数据集的二分类任务，直观对比从传统静态词向量到现代大语言模型（LLM）的架构演进与性能差异。

## 🚀 核心特性
* **五大核心算法**：横向部署并评测了 Word2Vec+LSTM, BERT, GPT, XLNet, RoBERTa。
* **分层抽样**：严格保证训练集与测试集分布一致，拒绝“数据虚高”。
* **全自动化评估**：一键跑通所有模型，并自动生成 Precision/Recall/F1-Score 多维度对比图表。
* **Gradio 可视化系统**：集成所有模型权重，提供 Web UI 界面进行多模型联合实时判定。
* **对抗测试设计**：针对模型易产生的“捷径学习（Shortcut Learning）”设计了 Hard Ham 边缘样本。

## 📂 项目结构
```text
├── data/                       # 数据集目录
├── lstm_saved_model.pkl        # LSTM 序列化权重与词表
├── word2vec_lstm.py            # 传统架构基线代码
├── transformer_models.py       # 四大预训练模型全自动微调与画图引擎
├── app.py                      # Gradio 多模型可视化 Web 界面
├── requirements.txt            # 环境依赖清单
└── README.md