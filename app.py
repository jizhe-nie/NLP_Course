import gradio as gr
import torch
import torch.nn as nn
import jieba
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ==========================================
# 1. 定义 LSTM 模型结构 (用于加载权重)
# ==========================================
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128):
        super(LSTMClassifier, self).__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden_cat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        return self.fc(hidden_cat)


# ==========================================
# 2. 预加载所有模型
# ==========================================
print("正在将所有模型加载至内存，请稍候...")
models = {}
tokenizers = {}

# --- 加载 LSTM ---
try:
    with open('lstm_saved_model.pkl', 'rb') as f:
        lstm_data = pickle.load(f)
    lstm_word2idx = lstm_data['word2idx']
    lstm_model = LSTMClassifier(lstm_data['embedding_matrix'])
    lstm_model.load_state_dict(lstm_data['state_dict'])
    lstm_model.eval()
    print("✅ Word2Vec+LSTM 模型加载成功!")
except Exception as e:
    print(f"⚠️ LSTM 模型加载失败，请确保先运行并保存了 lstm_saved_model.pkl。错误: {e}")
    lstm_model = None

# --- 加载 Transformer 大模型 ---
# 本地模型权重路径
paths = {
    "BERT": "./results_BERT/checkpoint-750",
    "GPT": "./results_GPT/checkpoint-750",
    "XLNet": "./results_XLNet/checkpoint-750",
    "RoBERTa": "./results_RoBERTa/checkpoint-750"
}

# 强制指定官方的 Tokenizer 字典名称，防止本地文件夹字典丢失导致乱码！
original_names = {
    "BERT": "bert-base-chinese",
    "GPT": "uer/gpt2-chinese-cluecorpussmall",
    "XLNet": "hfl/chinese-xlnet-base",
    "RoBERTa": "hfl/chinese-roberta-wwm-ext"
}

for name, path in paths.items():
    try:
        # 【关键修复】分词器从官方原始名字加载，确保字典 100% 正确
        tokenizers[name] = AutoTokenizer.from_pretrained(original_names[name])
        # 模型权重依然从你本地训练好的 checkpoint 文件夹加载
        models[name] = AutoModelForSequenceClassification.from_pretrained(path)
        models[name].eval()
        print(f"✅ {name} 模型加载成功!")
    except Exception as e:
        print(f"⚠️ {name} 模型加载失败，请检查路径。错误信息: {e}")


# ==========================================
# 3. 定义核心多模型研判函数
# ==========================================
def predict_email(text):
    if not text.strip():
        return "请输入邮件内容"

    results = {}

    # --------------------------------
    # 评委 1：传统架构 LSTM 进行预测
    # --------------------------------
    if lstm_model is not None:
        # 使用 jieba 分词并转为索引
        encoded = [lstm_word2idx.get(word, 1) for word in jieba.cut(text)]  # 1 是 UNK 的索引
        max_len = 100
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))  # 0 是 PAD 的索引
        else:
            encoded = encoded[:max_len]

        lstm_input = torch.LongTensor([encoded])

        with torch.no_grad():
            outputs = lstm_model(lstm_input)
            probs = torch.softmax(outputs, dim=1)[0]
            spam_prob = probs[1].item()

            if spam_prob > 0.5:
                results["Word2Vec+LSTM"] = f"🚫 垃圾邮件 (置信度: {spam_prob:.2%})"
            else:
                results["Word2Vec+LSTM"] = f"✅ 正常邮件 (置信度: {1 - spam_prob:.2%})"
    else:
        results["Word2Vec+LSTM"] = "⚠️ 模型未加载"

    # --------------------------------
    # 评委 2-5：四大大模型进行预测
    # --------------------------------
    for name in ["BERT", "GPT", "XLNet", "RoBERTa"]:
        if name in models and models[name] is not None:
            tokenizer = tokenizers[name]
            model = models[name]

            if "gpt" in name.lower() and tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '[PAD]'

            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[0]
                spam_prob = probs[1].item()

                if spam_prob > 0.5:
                    results[name] = f"🚫 垃圾邮件 (置信度: {spam_prob:.2%})"
                else:
                    results[name] = f"✅ 正常邮件 (置信度: {1 - spam_prob:.2%})"
        else:
            results[name] = "⚠️ 模型未加载"

    # --------------------------------
    # 格式化输出最终报告
    # --------------------------------
    output_str = f"【文本分析目标】\n{text}\n\n【五大架构模型研判结果】\n"
    output_str += "=" * 40 + "\n"
    for name, res in results.items():
        # 让输出格式更整齐
        output_str += f"[{name.ljust(13)}] : {res}\n"
        output_str += "-" * 40 + "\n"

    return output_str


# ==========================================
# 4. 构建前端界面
# ==========================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ 多架构垃圾邮件联合拦截检测系统")
    gr.Markdown(
        "本系统后端集成了**传统循环神经网络 (Word2Vec+LSTM)** 与 **四大预训练 Transformer 模型 (BERT, GPT, XLNet, RoBERTa)**，全面展示不同 NLP 发展阶段架构的判定逻辑与置信度差异。")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=8,
                placeholder="请输入需要检测的邮件内容...",
                label="输入测试文本"
            )
            submit_btn = gr.Button("🚀 提交联合检测", variant="primary")

            with gr.Row():
                gr.Examples(
                    examples=[
                        "周总您好，附件是本月研发团队的绩效考核报表，请查阅批示。另外下午3点在第一会议室有部门例会。",
                        "您好！我公司实力雄厚，专业代开各行业增值税普通/专用发票，税率优惠，绝对保真，支持上网查验后付款。联系电话：138xxxx",
                        "澳门皇家赌场上线啦！性感荷官在线发牌，首充送1888元体验金，点击链接立即暴富：http://xxx.com",
                        "您好，关于上周订购的步进电机和微距镜头，配件已经全部收到。附件是这批设备的增值税专用发票扫描件，请查收。另外，相关的运费报销单我已经提交给财务审核了，麻烦您跟进一下。祝好！",
                        "您好，上周订购的步进电机和微距镜头，配件已经全部收到。附件是这批设备的技术规格书扫描件，请查收。另外，相关的运费报销单我已经提交给财务审核了，麻烦您跟进一下。祝好！"
                    ],
                    inputs=input_text,
                    label="典型对抗测试用例"
                )

        with gr.Column():
            output_display = gr.Textbox(lines=18, label="全模型融合判决报告")

    submit_btn.click(fn=predict_email, inputs=input_text, outputs=output_display)

if __name__ == "__main__":
    demo.launch(share=False)