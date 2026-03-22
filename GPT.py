import os
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# 禁用由于并行分词器带来的一些无害警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==========================================
# 1. 统一的数据加载 (不用jieba分词了)
# ==========================================
def load_data(ham_path, spam_path):
    print("正在加载数据...")
    texts, labels = [], []

    with open(ham_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts.extend([line.strip() for line in f if line.strip()])
        labels.extend([0] * len(texts))

    spam_start_idx = len(texts)
    with open(spam_path, 'r', encoding='utf-8', errors='ignore') as f:
        texts.extend([line.strip() for line in f if line.strip()])
        labels.extend([1] * (len(texts) - spam_start_idx))

    df = pd.DataFrame({'text': texts, 'label': labels})

    # 采用完全相同的随机种子(42)，确保训练集和测试集与 LSTM 实验一模一样
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
# 3. BERT 模型训练与评估主流程
# ==========================================
def main():
    ham_path = 'data/ham_data.txt'
    spam_path = 'data/spam_data.txt'

    train_df, test_df = load_data(ham_path, spam_path)

    # 转换为 Hugging Face 专用的 Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 定义要使用的预训练模型 (这里使用经典的中文 BERT)
    # 中文小参数量 GPT2
    model_name = "uer/gpt2-chinese-cluecorpussmall"
    print(f"\n正在加载预训练模型和分词器: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ====== 核心修复区：强制为 GPT 注入 PAD Token ======
    # 1. 如果分词器没有 pad_token
    if tokenizer.pad_token is None:
        # 尝试使用 eos_token 替代
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # 如果连 eos_token 都没有，就强行向字典里添加一个专属的 '[PAD]' 字符
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 2. 关键一步：必须把 tokenizer 新认定的 pad_token_id 同步给 model 的底层配置
    model.config.pad_token_id = tokenizer.pad_token_id

    # 3. 如果我们刚才强行添加了新词 '[PAD]'，必须让模型底层的词表矩阵扩容，否则会引发张量越界报错
    model.resize_token_embeddings(len(tokenizer))

    # ====================================================

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    print("正在对数据集进行 Tokenize 编码...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    test_tokenized = test_dataset.map(tokenize_function, batched=True)

    # 设定训练参数
    training_args = TrainingArguments(
        output_dir="./results_bert",
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        eval_strategy="epoch",  # <--- 修复：这里的参数名改成了 eval_strategy
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        logging_dir='./logs_bert'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )

    print("\n开始训练 BERT 模型...")
    trainer.train()

    print("\n生成测试集评估报告...")
    eval_results = trainer.evaluate()

    print("-" * 30)
    print("【GPT 模型评估报告】")
    print(f"准确率 (Accuracy) : {eval_results['eval_accuracy']:.4f}")
    print(f"精确率 (Precision): {eval_results['eval_precision']:.4f}")
    print(f"召回率 (Recall)   : {eval_results['eval_recall']:.4f}")
    print(f"F1 Score        : {eval_results['eval_f1']:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()