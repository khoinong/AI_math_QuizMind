# -*- coding: utf-8 -*-
"""Testing script cho mô hình NER với định dạng CoNLL"""

import os
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Cấu hình
# =========================
MODEL_DIR = "src/model/ner_model"
TEST_FILE = "data/test_ner.conll"  # Đường dẫn đến file test
OUTPUT_DIR = "src/model/ner_model/test_results"
BATCH_SIZE = 16

# Định nghĩa các nhãn (phải giống với training)
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

# =========================
# Hàm đọc file CoNLL
# =========================
def read_conll_file(file_path):
    """Đọc file CoNLL và trả về tokens và labels"""
    tokens = []
    labels = []
    current_tokens = []
    current_labels = []

    try:
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
            
            # Thêm câu cuối cùng nếu có
            if current_tokens:
                tokens.append(current_tokens)
                labels.append(current_labels)
                
        print(f"✓ Đọc được {len(tokens)} câu từ file {file_path}")
        return tokens, labels
        
    except FileNotFoundError:
        print(f"✗ Không tìm thấy file {file_path}")
        return [], []

# =========================
# Hàm xử lý tokenize và align labels
# =========================
def tokenize_and_align_labels(examples, tokenizer):
    """Tokenize và căn chỉnh nhãn cho dữ liệu test"""
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
            if len(word_tokens) > 1:
                label_ids.extend([-100] * (len(word_tokens) - 1))

        input_ids.append(tokenizer.sep_token_id)
        attention_mask.append(1)
        label_ids.append(-100)

        # Cắt bớt nếu vượt quá 512 tokens
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

# =========================
# Hàm visualization
# =========================
def plot_confusion_matrix(true_labels, pred_labels, label_list, save_path):
    """Vẽ confusion matrix cho NER"""
    from seqeval.metrics import accuracy_score
    
    # Tạo confusion matrix đơn giản
    unique_labels = sorted(list(set([label for seq in true_labels for label in seq])))
    
    # Tính số lượng xuất hiện của mỗi cặp true-pred
    confusion_data = {}
    for true_seq, pred_seq in zip(true_labels, pred_labels):
        for true_label, pred_label in zip(true_seq, pred_seq):
            key = (true_label, pred_label)
            confusion_data[key] = confusion_data.get(key, 0) + 1
    
    # Tạo matrix
    cm = np.zeros((len(unique_labels), len(unique_labels)))
    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            cm[i, j] = confusion_data.get((true_label, pred_label), 0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title('Confusion Matrix - NER')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_metrics(report_dict, save_path):
    """Vẽ biểu đồ hiệu suất cho từng entity"""
    entities = [label for label in report_dict.keys() if label not in ['micro avg', 'macro avg', 'weighted avg', 'accuracy']]
    
    precisions = [report_dict[ent]['precision'] for ent in entities]
    recalls = [report_dict[ent]['recall'] for ent in entities]
    f1_scores = [report_dict[ent]['f1-score'] for ent in entities]
    
    x = np.arange(len(entities))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precisions, width, label='Precision', alpha=0.7)
    plt.bar(x, recalls, width, label='Recall', alpha=0.7)
    plt.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.7)
    
    plt.xlabel('Entity Types')
    plt.ylabel('Score')
    plt.title('Performance Metrics by Entity Type')
    plt.xticks(x, entities, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_entity_distribution(true_labels, pred_labels, save_path):
    """Vẽ biểu đồ phân bố entity"""
    true_entities = [label for seq in true_labels for label in seq if label != 'O']
    pred_entities = [label for seq in pred_labels for label in seq if label != 'O']
    
    true_counts = {}
    pred_counts = {}
    
    for entity in true_entities:
        true_counts[entity] = true_counts.get(entity, 0) + 1
    
    for entity in pred_entities:
        pred_counts[entity] = pred_counts.get(entity, 0) + 1
    
    # Chuẩn bị dữ liệu cho biểu đồ
    all_entities = sorted(set(list(true_counts.keys()) + list(pred_counts.keys())))
    true_values = [true_counts.get(entity, 0) for entity in all_entities]
    pred_values = [pred_counts.get(entity, 0) for entity in all_entities]
    
    x = np.arange(len(all_entities))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, true_values, width, label='Thực tế', alpha=0.7)
    plt.bar(x + width/2, pred_values, width, label='Dự đoán', alpha=0.7)
    
    plt.xlabel('Entity Types')
    plt.ylabel('Số lượng')
    plt.title('Phân bố Entity - Thực tế vs Dự đoán')
    plt.xticks(x, all_entities, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# =========================
# Hàm chính
# =========================
def main():
    print("=== BẮT ĐẦU TESTING MÔ HÌNH NER ===\n")
    
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load dữ liệu test
    print("[1] Đang load dữ liệu test...")
    test_tokens, test_labels = read_conll_file(TEST_FILE)
    
    if not test_tokens:
        print("✗ Không có dữ liệu test để đánh giá")
        return
    
    # 2. Tạo dataset
    print("[2] Đang tạo dataset...")
    test_dataset = Dataset.from_dict({
        'tokens': test_tokens,
        'ner_tags': [[label2id[label] for label in seq] for seq in test_labels]
    })
    
    # 3. Load model và tokenizer
    print("[3] Đang load model và tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=False)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        print("✓ Load model thành công")
    except Exception as e:
        print(f"✗ Lỗi khi load model: {e}")
        return
    
    # 4. Tokenize dữ liệu test
    print("[4] Đang tokenize dữ liệu...")
    def tokenize_function(examples):
        return tokenize_and_align_labels(examples, tokenizer)
    
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # 5. Dự đoán với PyTorch
    print("[5] Đang dự đoán...")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_label_ids = []
    
    with torch.no_grad():
        for i in range(0, len(tokenized_test), BATCH_SIZE):
            batch_end = min(i + BATCH_SIZE, len(tokenized_test))
            batch_data = tokenized_test[i:batch_end]
            
            # Pad sequences to the same length
            max_len = max(len(seq) for seq in batch_data['input_ids'])
            
            padded_input_ids = []
            padded_attention_mask = []
            padded_labels = []
            
            for input_id, attention_mask, label_seq in zip(batch_data['input_ids'], batch_data['attention_mask'], batch_data['labels']):
                # Pad with 0s for input_ids and attention_mask
                pad_len = max_len - len(input_id)
                padded_input_ids.append(input_id + [tokenizer.pad_token_id] * pad_len)
                padded_attention_mask.append(attention_mask + [0] * pad_len)
                # Pad with -100 for labels (ignore index)
                padded_labels.append(label_seq + [-100] * pad_len)
            
            input_ids = torch.tensor(padded_input_ids).to(device)
            attention_mask = torch.tensor(padded_attention_mask).to(device)
            labels = torch.tensor(padded_labels)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=2)
            all_preds.extend(preds.cpu().numpy())
            all_label_ids.extend(labels.numpy())
    
    preds = np.array(all_preds)
    
    # 6. Xử lý kết quả
    print("[6] Đang xử lý kết quả...")
    true_labels = []
    pred_labels = []
    
    for i, label_seq in enumerate(all_label_ids):
        true_line = []
        pred_line = []
        for j, label_id in enumerate(label_seq):
            if label_id != -100:  # Chỉ lấy các token thực sự (không phải special tokens)
                true_line.append(id2label[label_id])
                pred_line.append(id2label[preds[i][j]])
        
        # Chỉ thêm các sequence không rỗng
        if true_line:
            true_labels.append(true_line)
            pred_labels.append(pred_line)
    
    # 8. Tính toán metrics
    print("[7] Đang tính toán độ chính xác...")
    
    # Sử dụng seqeval cho đánh giá NER
    report = classification_report(true_labels, pred_labels, output_dict=True)
    
    # Tính các chỉ số tổng quan
    micro_f1 = f1_score(true_labels, pred_labels, average='micro')
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    weighted_f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    precision = precision_score(true_labels, pred_labels, average='micro')
    recall = recall_score(true_labels, pred_labels, average='micro')
    
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH NER")
    print("="*50)
    print(f"Tổng số câu test: {len(true_labels)}")
    print(f"Tổng số tokens: {sum(len(seq) for seq in true_labels)}")
    print(f"Micro F1-Score: {micro_f1:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Weighted F1-Score: {weighted_f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # 9. In báo cáo chi tiết
    print("\n" + "="*50)
    print("BÁO CÁO CHI TIẾT THEO ENTITY")
    print("="*50)
    print(classification_report(true_labels, pred_labels))
    
    # 10. Lưu kết quả chi tiết
    print("[8] Đang lưu kết quả chi tiết...")
    
    # Tạo DataFrame kết quả
    results = []
    for i, (true_seq, pred_seq, tokens) in enumerate(zip(true_labels, pred_labels, test_tokens)):
        for j, (true_label, pred_label, token) in enumerate(zip(true_seq, pred_seq, tokens)):
            results.append({
                'sentence_id': i,
                'token': token,
                'true_label': true_label,
                'pred_label': pred_label,
                'is_correct': true_label == pred_label
            })
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(OUTPUT_DIR, "ner_test_results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Kết quả chi tiết lưu tại: {results_csv_path}")
    
    # 11. Visualization
    print("[9] Đang tạo biểu đồ...")
    
    # Confusion Matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(true_labels, pred_labels, label_list, cm_path)
    print(f"✓ Confusion matrix lưu tại: {cm_path}")
    
    # Performance metrics by entity
    metrics_path = os.path.join(OUTPUT_DIR, "performance_metrics.png")
    plot_performance_metrics(report, metrics_path)
    print(f"✓ Biểu đồ hiệu suất lưu tại: {metrics_path}")
    
    # Entity distribution
    dist_path = os.path.join(OUTPUT_DIR, "entity_distribution.png")
    plot_entity_distribution(true_labels, pred_labels, dist_path)
    print(f"✓ Biểu đồ phân bố entity lưu tại: {dist_path}")
    
    # 12. Phân tích lỗi
    print("[10] Phân tích lỗi...")
    
    # Tính accuracy theo từng entity type
    entity_accuracy = {}
    for entity in set([label for seq in true_labels for label in seq]):
        if entity == 'O':
            continue
        
        entity_true = [label for seq in true_labels for label in seq if label == entity]
        entity_pred = [pred for true, pred in zip([label for seq in true_labels for label in seq], 
                                                 [label for seq in pred_labels for label in seq]) 
                      if true == entity]
        
        if entity_true:
            entity_acc = sum(1 for t, p in zip(entity_true, entity_pred) if t == p) / len(entity_true)
            entity_accuracy[entity] = entity_acc
    
    print("\n" + "="*50)
    print("PHÂN TÍCH ĐỘ CHÍNH XÁC THEO ENTITY")
    print("="*50)
    for entity, acc in sorted(entity_accuracy.items()):
        print(f"{entity}: {acc:.2%}")
    
    # Phân tích các lỗi phổ biến
    error_analysis = results_df[results_df['is_correct'] == False]
    if len(error_analysis) > 0:
        common_errors = error_analysis.groupby(['true_label', 'pred_label']).size().reset_index(name='count')
        common_errors = common_errors.sort_values('count', ascending=False)
        
        print(f"\nCÁC LỖI DỰ ĐOÁN PHỔ BIẾN (Top 10):")
        for _, row in common_errors.head(10).iterrows():
            print(f"  {row['true_label']} → {row['pred_label']}: {row['count']} lần")
    
    # 13. Lưu báo cáo tổng hợp
    summary_report = {
        'total_sentences': len(true_labels),
        'total_tokens': sum(len(seq) for seq in true_labels),
        'overall_accuracy': sum(results_df['is_correct']) / len(results_df),
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'precision': precision,
        'recall': recall,
        'entity_accuracy': entity_accuracy
    }
    
    with open(os.path.join(OUTPUT_DIR, "summary_report.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*50)
    print("HOÀN TẤT TESTING")
    print("="*50)
    print(f"✓ Tất cả kết quả được lưu trong: {OUTPUT_DIR}")
    print(f"✓ Độ chính xác tổng thể: {summary_report['overall_accuracy']:.2%}")
    print(f"✓ Micro F1-Score: {micro_f1:.4f}")

if __name__ == "__main__":
    main()