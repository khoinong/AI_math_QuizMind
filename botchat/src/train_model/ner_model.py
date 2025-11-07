# -*- coding: utf-8 -*-
"""Huấn luyện mô hình NER với định dạng CoNLL"""

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    TrainerCallback
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import classification_report
from tqdm.auto import tqdm   # <-- thêm tqdm

# Định nghĩa các nhãn
label_list = [
    "O",
    "B-NUM", "I-NUM",
    "B-AGENT", "I-AGENT",
    "B-REL", "I-REL",
    "B-VALUE", "I-VALUE",
    "B-UNIT", "I-UNIT",
    "B-ATTRIBUTE", "I-ATTRIBUTE",
    "B-QUESTION", "I-QUESTION"
]

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# Đọc và xử lý dữ liệu từ file CoNLL
def read_conll_file(file_path):
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[0]
                    label = parts[-1]
                    current_tokens.append(token)
                    current_labels.append(label)
            else:
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
        if current_tokens:
            tokens.append(current_tokens)
            labels.append(current_labels)
    return tokens, labels

# Đường dẫn đến tập train/validation/test
train_tokens, train_labels = read_conll_file("data/train.conll")
val_tokens, val_labels = read_conll_file("data/dev.conll")

# Tạo dataset
def create_dataset(tokens, labels):
    return Dataset.from_dict({
        'tokens': tokens,
        'ner_tags': [[label2id[label] for label in seq] for seq in labels]
    })

train_dataset = create_dataset(train_tokens, train_labels)
val_dataset = create_dataset(val_tokens, val_labels)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2", use_fast=False)

# Tokenize và căn chỉnh nhãn
def tokenize_and_align_labels(examples):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for tokens, labels in zip(examples["tokens"], examples["ner_tags"]):
        input_ids = [tokenizer.cls_token_id]
        label_ids = [-100]
        attention_mask = [1]

        for word, label in zip(tokens, labels):
            word_tokens = tokenizer.tokenize(word)
            word_token_ids = tokenizer.convert_tokens_to_ids(word_tokens)

            input_ids.extend(word_token_ids)
            attention_mask.extend([1] * len(word_token_ids))

            label_ids.append(label)
            if len(word_token_ids) > 1:
                label_ids.extend([-100] * (len(word_token_ids) - 1))

        input_ids.append(tokenizer.sep_token_id)
        attention_mask.append(1)
        label_ids.append(-100)

        if len(input_ids) > 512:
            input_ids = input_ids[:512]
            attention_mask = attention_mask[:512]
            label_ids = label_ids[:512]

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(label_ids)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels,
    }

# Áp dụng xử lý tokenize
tokenized_train = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_val = val_dataset.map(tokenize_and_align_labels, batched=True)

# Data collator
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    "vinai/phobert-base-v2",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="src/model/ner_model",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy="epoch",
    save_total_limit=2,
    max_grad_norm=1.0,
    logging_dir="output/logs",
    logging_steps=10,
    load_best_model_at_end=True,
    no_cuda=True,
)

# Callback để thêm thanh progress bar bằng tqdm

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,

)

# Huấn luyện mô hình
trainer.train(resume_from_checkpoint=False)

# Đánh giá mô hình
predictions = trainer.predict(tokenized_val)
preds = np.argmax(predictions.predictions, axis=2)

true_labels = []
pred_labels = []
for i, label_seq in enumerate(predictions.label_ids):
    true_line = []
    pred_line = []
    for j, label_id in enumerate(label_seq):
        if label_id != -100:
            true_line.append(id2label[label_id])
            pred_line.append(id2label[preds[i][j]])
    true_labels.append(true_line)
    pred_labels.append(pred_line)

print(classification_report(true_labels, pred_labels))

# Lưu mô hình
trainer.save_model("src/model/ner_model")
tokenizer.save_pretrained("src/model/ner_model")
