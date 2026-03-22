import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 禁用分词器并行带来的无关警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 解决 Mac 系统下 matplotlib 无法显示中文字体的问题
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac 自带的中文字体
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


# ==========================================
# 1. 统一数据加载模块
# ==========================================
def load_data(ham_path, spam_path):
    print("正在加载数据集...")
    texts, labels = [], []

    with open(ham_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts.extend([line.strip() for line in f if line.strip()])
        labels.extend([0] * len(texts))

    spam_start_idx = len(texts)
    with open(spam_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts.extend([line.strip() for line in f if line.strip()])
        labels.extend([1] * (len(texts) - spam_start_idx))

    df = pd.DataFrame({'text': texts, 'label': labels})

    # 完全相同的随机种子，确保测试集公平
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    return train_df, test_df


# ==========================================
# 2. 评价指标计算
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }


# ==========================================
# 3. 自动化图表生成模块
# ==========================================
def generate_comparison_charts(results_df):
    print("\n正在生成指标对比图像并保存到本地...")

    # 统一设置图表样式
    sns.set_theme(style="whitegrid")
    try:
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']

    # 1. 生成各单个指标的对比图 (4张图)
    for metric, color in zip(metrics, colors):
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x='Model', y=metric, data=results_df, color=color, width=0.6)

        plt.title(f'{metric} 跨模型对比', fontsize=14)
        plt.ylabel(f'{metric} Score', fontsize=12)
        plt.ylim(0.985, 1.002)  # 为了放大差异，将Y轴截断

        # 在柱子上添加数据标签
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.4f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 8),
                        textcoords='offset points')

        plt.tight_layout()
        plt.savefig(f'Chart_{metric}_Comparison.png', dpi=300)
        plt.close()

    # 2. 生成所有指标与所有模型的综合对比直方图 (1张总图)
    results_melted = pd.melt(results_df, id_vars=['Model'], value_vars=metrics,
                             var_name='Metric', value_name='Score')

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Model', y='Score', hue='Metric', data=results_melted)

    plt.title('全架构模型性能综合对比', fontsize=16)
    plt.ylabel('Scores', fontsize=12)
    plt.ylim(0.985, 1.002)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('Chart_All_Metrics_Comparison.png', dpi=300)
    plt.close()

    print("图像生成完毕！已存入当前目录。")


# ==========================================
# 4. 主控引擎：自动跑通所有模型
# ==========================================
def main():
    ham_path = 'data/ham_data.txt'
    spam_path = 'data/spam_data.txt'

    train_df, test_df = load_data(ham_path, spam_path)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 将你之前的 LSTM 成绩硬编码进来，方便一起画图
    final_results = [
        {"Model": "Word2Vec+LSTM", "Accuracy": 0.9915, "Precision": 0.9910, "Recall": 0.9920, "F1-Score": 0.9915}
    ]

    # 定义你要自动化运行的模型列表
    models_to_run = [
        ("BERT", "bert-base-chinese"),
        ("GPT", "uer/gpt2-chinese-cluecorpussmall"),
        ("XLNet", "hfl/chinese-xlnet-base"),
        ("RoBERTa", "hfl/chinese-roberta-wwm-ext")
    ]

    # 开始自动化循环
    for short_name, model_path in models_to_run:
        print("\n" + "=" * 50)
        print(f"正在全自动处理模型: 【{short_name}】")
        print("=" * 50)

        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 为 GPT 的基因缺陷打上自动补丁
        if "gpt" in model_path.lower():
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

        # 同步 GPT 的补丁到模型底层
        if "gpt" in model_path.lower():
            model.config.pad_token_id = tokenizer.pad_token_id
            model.resize_token_embeddings(len(tokenizer))

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

        print("进行 Tokenize 编码...")
        train_tokenized = train_dataset.map(tokenize_function, batched=True)
        test_tokenized = test_dataset.map(tokenize_function, batched=True)

        training_args = TrainingArguments(
            output_dir=f"./results_{short_name}",
            num_train_epochs=3,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=64,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            load_best_model_at_end=True,
            logging_dir=f'./logs_{short_name}'
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=test_tokenized,
            compute_metrics=compute_metrics,
        )

        print(f"开始训练 {short_name} 模型...")
        trainer.train()

        print(f"评估 {short_name} 模型...")
        eval_metrics = trainer.evaluate()

        # 记录该模型的成绩
        final_results.append({
            "Model": short_name,
            "Accuracy": eval_metrics['eval_accuracy'],
            "Precision": eval_metrics['eval_precision'],
            "Recall": eval_metrics['eval_recall'],
            "F1-Score": eval_metrics['eval_f1']
        })

        print(f"【{short_name}】处理完毕！成绩已收录。")

    # 所有循环结束后，整合数据
    print("\n" + "=" * 50)
    print("所有预训练语言模型实验全部完成！")
    print("=" * 50)

    results_df = pd.DataFrame(final_results)
    print("\n最终指标对比矩阵：")
    print(results_df.to_string(index=False))

    # 自动生成所有图像
    generate_comparison_charts(results_df)


if __name__ == "__main__":
    main()