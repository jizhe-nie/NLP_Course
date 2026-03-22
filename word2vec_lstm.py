import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np


# ==========================================
# 1. 数据加载与中文分词
# ==========================================
def load_and_tokenize_data(ham_path, spam_path):
    print("正在加载数据并进行中文分词 (这可能需要一两分钟)...")
    texts, labels = [], []

    # 读取正常邮件 (标签 0)
    with open(ham_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(list(jieba.cut(line)))  # jieba分词
                labels.append(0)

    # 读取垃圾邮件 (标签 1)
    with open(spam_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(list(jieba.cut(line)))
                labels.append(1)

    # 划分数据集：80%训练集, 20%测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_train, X_test, y_train, y_test


# ==========================================
# 2. 训练 Word2Vec 与 构建词表
# ==========================================
def build_word2vec_and_vocab(X_train, vector_size=100):
    print("正在训练 Word2Vec 词向量...")
    # 只在训练集上训练词向量，防止数据泄露
    w2v_model = Word2Vec(sentences=X_train, vector_size=vector_size, window=5, min_count=2, workers=4)

    # 构建词表映射 (词 -> 索引)
    word2idx = {"<PAD>": 0, "<UNK>": 1}  # PAD用于填充短句子，UNK用于处理未登录词
    embedding_matrix = [np.zeros(vector_size), np.random.randn(vector_size)]  # 对应的初始向量

    for word in w2v_model.wv.index_to_key:
        word2idx[word] = len(word2idx)
        embedding_matrix.append(w2v_model.wv[word])

    embedding_matrix = np.array(embedding_matrix)
    return word2idx, torch.FloatTensor(embedding_matrix)


# ==========================================
# 3. 构建 PyTorch 数据集 (将文本转为数字索引)
# ==========================================
class SpamDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=100):
        self.labels = labels
        self.encoded_texts = []

        for text in texts:
            # 将汉字词语转化为数字索引，如果不认识的词就给 <UNK> 的索引(1)
            encoded = [word2idx.get(word, 1) for word in text]
            # 截断长句子或填充短句子到固定长度 max_len
            if len(encoded) < max_len:
                encoded += [0] * (max_len - len(encoded))  # 0 是 <PAD> 的索引
            else:
                encoded = encoded[:max_len]
            self.encoded_texts.append(encoded)

        self.encoded_texts = torch.LongTensor(self.encoded_texts)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.encoded_texts[idx], self.labels[idx]


# ==========================================
# 4. 定义 LSTM 神经网络模型
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128):
        super(LSTMClassifier, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape

        # 嵌入层：加载我们刚才训练好的 Word2Vec 向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False  # 冻结词向量，这里先不参与训练微调

        # 双向 LSTM 层
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 全连接分类层 (二分类)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        # x 形状: (batch_size, max_len)
        embedded = self.embedding(x)  # 形状: (batch_size, max_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)

        # 提取双向LSTM最后一个时间步的隐藏状态
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden_cat)


# ==========================================
# 5. 主控程序：训练与评估
# ==========================================
def main():
    # 设定路径
    ham_path = 'data/ham_data.txt'
    spam_path = 'data/spam_data.txt'

    # 1. 准备数据
    X_train, X_test, y_train, y_test = load_and_tokenize_data(ham_path, spam_path)

    # 2. 词向量与词表
    word2idx, embedding_matrix = build_word2vec_and_vocab(X_train)

    # 3. 封装为 DataLoader
    train_dataset = SpamDataset(X_train, y_train, word2idx)
    test_dataset = SpamDataset(X_test, y_test, word2idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 4. 初始化模型、损失函数和优化器
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"正在使用的计算设备: {device}")

    model = LSTMClassifier(embedding_matrix).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 5. 开始训练
    epochs = 5
    print("开始训练 LSTM 模型...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1} 训练平均Loss: {total_loss / len(train_loader):.4f}")

    # 6. 测试集评估与计算指标
    print("\n开始在测试集上评估并计算四个评价指标...")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # 计算你课程要求的指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("-" * 30)
    print("【Word2Vec + LSTM 评估报告】")
    print(f"准确率 (Accuracy) : {acc:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall)   : {recall:.4f}")
    print(f"F1 Score        : {f1:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()