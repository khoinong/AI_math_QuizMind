# -*- coding: utf-8 -*-
"""Huấn luyện mô hình tách câu"""

from dataclasses import dataclass
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
    TrainingArguments, 
    Trainer
)
from datasets import load_dataset
from tqdm.auto import tqdm
import transformers

# Thiết lập logging
transformers.logging.set_verbosity_error()

# Cấu hình model
MODEL_NAME = "vinai/phobert-base-v2"  # Sử dụng phiên bản mới nhất của PhoBERT
LABELS = ["CONT", "BREAK"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

# Load fast tokenizer
from transformers import RobertaTokenizerFast
try:
    # Thử tải fast tokenizer đã lưu nếu có
    tokenizer = RobertaTokenizerFast.from_pretrained("src/model/fast_tokenizer")
except:
    print("Tải fast tokenizer từ PhoBERT...")
    tokenizer = RobertaTokenizerFast.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        add_prefix_space=True,
        model_max_length=512
    )
    # Lưu fast tokenizer để sử dụng lại
    tokenizer.save_pretrained("src/model/fast_tokenizer")

print(f"Tokenizer type: {type(tokenizer)}")
if not tokenizer.is_fast:
    raise ValueError("Cần sử dụng fast tokenizer cho word_ids()")

LABELS = ["CONT","BREAK"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

def build_labels_for_example(text, word_break_indices):
    """Xây dựng nhãn cho từng từ trong câu.
    
    Args:
        text: Chuỗi văn bản đầu vào
        word_break_indices: Set các vị trí từ (0-based) có BREAK ngay sau
    
    Returns:
        dict: Kết quả tokenize với nhãn cho từng token
    """
    # Tách từ và tokenize
    words = text.split()
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        return_offsets_mapping=True,
        padding=False,
    )
    
    # Khởi tạo nhãn
    labels = np.full(len(encoding.input_ids), -100, dtype=int)
    
    # Lấy word_ids cho mỗi token
    word_ids = encoding.word_ids()
    
    # Duyệt qua các token và gán nhãn
    for i in range(len(encoding.input_ids)):
        word_idx = word_ids[i]
        
        # Bỏ qua special tokens
        if word_idx is None:
            continue
            
        # Kiểm tra xem đây có phải token cuối của từ không
        is_last_token = (i == len(encoding.input_ids)-1) or (word_ids[i+1] != word_idx)
        
        if is_last_token:
            # Gán nhãn BREAK nếu từ này kết thúc câu
            label = "BREAK" if word_idx in word_break_indices else "CONT"
            labels[i] = label2id[label]
    
    # Trả về kết quả
    encoding["labels"] = labels.tolist()
    return encoding

def preprocess(batch):
    texts = batch["text"]
    breaks = batch["breaks"]
    out = {k:[] for k in ["input_ids","attention_mask","labels"]}
    for t, br in zip(texts, breaks):
        enc = build_labels_for_example(t, set(br))
        for k in out: out[k].append(enc[k])
    return out

# Expect JSONL with fields: text (string), breaks (list[int])
# Example generation step (offline): from sentences s1|s2|s3 -> text=" ".join(tokens_without_punct), breaks=[idx_of_last_token_s1, idx_last_s2, ...]
data = load_dataset("json", data_files={"train":"data/train_separate.jsonl","validation":"data/dev_separate.jsonl"})
data = data.map(preprocess, batched=True, remove_columns=data["train"].column_names)

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=len(LABELS), 
    id2label=id2label, 
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="src/model/sent_split_model",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,
    logging_dir="output/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    no_cuda=True,
)

from sklearn.metrics import precision_recall_fscore_support

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    pred_marks, gold_marks = [], []
    for pr, lb in zip(preds, labels):
        m = (lb!=-100)
        pred_marks.extend(pr[m])
        gold_marks.extend(lb[m])
    precision, recall, f1, _ = precision_recall_fscore_support(gold_marks, pred_marks, average="binary", pos_label=label2id["BREAK"])
    return {"precision":precision, "recall":recall, "f1":f1}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

# Huấn luyện mô hình
print("Bắt đầu huấn luyện...")
trainer.train()

# Đánh giá mô hình trên tập validation
print("\nĐánh giá trên tập validation:")
eval_result = trainer.evaluate()
print(eval_result)

# Lưu mô hình và tokenizer
print("\nLưu mô hình...")
trainer.save_model("src/model/sent_split_model")
tokenizer.save_pretrained("src/model/sent_split_model")
print("Hoàn thành!")

# -------- Inference --------
def split_sentences(raw_text, threshold=0.5):
    words = raw_text.split()
    enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**enc).logits[0]  # [seq, 2]
        probs = torch.softmax(logits, dim=-1)
    pieces, cur = [], []
    word_ids = enc.word_ids(batch_index=0)
    for i, w_id in enumerate(word_ids):
        if w_id is None: 
            continue
        token_str = tokenizer.convert_ids_to_tokens(int(enc["input_ids"][0,i]))
        # chỉ append ở subword đầu để tái tạo từ
        is_first_sub = (i==0) or (word_ids[i-1] != w_id)
        if is_first_sub:
            cur.append(words[w_id])
        # check break ở subword cuối
        is_last_sub = (i+1==len(word_ids)) or (word_ids[i+1]!=w_id)
        if is_last_sub:
            if probs[i, label2id["BREAK"]].item() >= threshold:
                pieces.append(" ".join(cur).strip())
                cur = []
    if cur:
        pieces.append(" ".join(cur).strip())
    return pieces

# Demo
print(split_sentences("lan có 3 quả táo, mẹ cho thêm 4 quả táo nữa. Lan có tất cả bao nhiêu quả táo"))
