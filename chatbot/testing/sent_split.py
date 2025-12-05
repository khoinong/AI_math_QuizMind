# -*- coding: utf-8 -*-
"""Testing script cho mô hình tách câu"""

import os
import json
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification
)
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Cấu hình
# =========================
MODEL_DIR = "src/model/sent_split_model"
TEST_FILE = "data/test_separate.jsonl"  # File test JSONL
OUTPUT_DIR = "src/model/sent_split_model/test_results"
BATCH_SIZE = 16

# Định nghĩa labels
LABELS = ["CONT", "BREAK"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

# =========================
# Hàm tiện ích
# =========================
def load_test_data(test_file):
    """Load dữ liệu test từ file JSONL"""
    try:
        dataset = load_dataset("json", data_files={"test": test_file})
        print(f"✓ Loaded {len(dataset['test'])} samples from {test_file}")
        return dataset
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return None

def predict_sentence_breaks(model, tokenizer, text, threshold=0.5):
    """Dự đoán vị trí ngắt câu cho một văn bản"""
    words = text.split()
    
    # Tokenize
    enc = tokenizer(
        words, 
        is_split_into_words=True, 
        return_tensors="pt", 
        truncation=True, 
        max_length=512
    )
    
    # Move to device
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    
    # Dự đoán
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits[0]
        probs = torch.softmax(logits, dim=-1)
    
    # Lấy word_ids để ánh xạ trở lại từ
    word_ids = enc.word_ids(batch_index=0)
    
    # Thu thập dự đoán cho từng từ
    word_predictions = []
    current_word_id = None
    
    for i, w_id in enumerate(word_ids):
        if w_id is None:
            continue
            
        # Chỉ xem xét token cuối cùng của mỗi từ
        is_last_subtoken = (i + 1 == len(word_ids)) or (word_ids[i + 1] != w_id)
        
        if is_last_subtoken:
            break_prob = probs[i, label2id["BREAK"]].item()
            prediction = 1 if break_prob >= threshold else 0
            word_predictions.append({
                'word': words[w_id],
                'word_id': w_id,
                'break_prob': break_prob,
                'prediction': prediction,
                'position': i
            })
    
    return word_predictions

def evaluate_model(model, tokenizer, test_dataset, threshold=0.5):
    """Đánh giá mô hình trên tập test"""
    all_true_breaks = []
    all_pred_breaks = []
    all_break_probs = []
    
    results = []
    
    for example in tqdm(test_dataset, desc="Evaluating"):
        text = example['text']
        true_breaks = set(example['breaks'])
        words = text.split()
        
        # Dự đoán
        word_predictions = predict_sentence_breaks(model, tokenizer, text, threshold)
        
        # Thu thập kết quả
        true_labels = []
        pred_labels = []
        
        for i, word in enumerate(words):
            # Tìm prediction cho từ này
            pred_info = next((p for p in word_predictions if p['word_id'] == i), None)
            
            if pred_info:
                true_label = 1 if i in true_breaks else 0
                pred_label = pred_info['prediction']
                
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                all_break_probs.append(pred_info['break_prob'])
                
                results.append({
                    'text': text,
                    'word': word,
                    'word_position': i,
                    'true_break': true_label,
                    'pred_break': pred_label,
                    'break_prob': pred_info['break_prob'],
                    'is_correct': true_label == pred_label
                })
        
        all_true_breaks.extend(true_labels)
        all_pred_breaks.extend(pred_labels)
    
    return all_true_breaks, all_pred_breaks, all_break_probs, results

def plot_confusion_matrix(true_labels, pred_labels, save_path):
    """Vẽ confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['CONT', 'BREAK'], 
                yticklabels=['CONT', 'BREAK'])
    plt.title('Confusion Matrix - Sentence Breaking')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(true_labels, break_probs, save_path):
    """Vẽ đường Precision-Recall"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    precision, recall, thresholds = precision_recall_curve(true_labels, break_probs)
    avg_precision = average_precision_score(true_labels, break_probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, linewidth=2, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_analysis(true_labels, break_probs, save_path):
    """Phân tích F1 theo các ngưỡng khác nhau"""
    thresholds = np.arange(0.1, 0.95, 0.05)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for threshold in thresholds:
        pred_labels = [1 if prob >= threshold else 0 for prob in break_probs]
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='binary', zero_division=0
        )
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # Tìm ngưỡng tốt nhất
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores, 'b-', marker='o', label='F1 Score', linewidth=2)
    plt.plot(thresholds, precision_scores, 'r-', marker='s', label='Precision', alpha=0.7)
    plt.plot(thresholds, recall_scores, 'g-', marker='^', label='Recall', alpha=0.7)
    
    # Đánh dấu điểm tốt nhất
    plt.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7,
                label=f'Best threshold: {best_threshold:.2f} (F1: {best_f1:.3f})')
    plt.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("F1 Score vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return best_threshold, best_f1

def plot_error_analysis(results_df, save_path):
    """Phân tích lỗi theo xác suất"""
    errors = results_df[results_df['is_correct'] == False]
    
    if len(errors) > 0:
        plt.figure(figsize=(12, 6))
        
        # Phân loại lỗi
        false_positives = errors[errors['true_break'] == 0]
        false_negatives = errors[errors['true_break'] == 1]
        
        plt.subplot(1, 2, 1)
        if len(false_positives) > 0:
            plt.hist(false_positives['break_prob'], bins=20, alpha=0.7, color='red', label='False Positives')
            plt.axvline(x=false_positives['break_prob'].mean(), color='darkred', linestyle='--', 
                       label=f'Mean: {false_positives["break_prob"].mean():.3f}')
        plt.xlabel('Break Probability')
        plt.ylabel('Count')
        plt.title('False Positives Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if len(false_negatives) > 0:
            plt.hist(false_negatives['break_prob'], bins=20, alpha=0.7, color='blue', label='False Negatives')
            plt.axvline(x=false_negatives['break_prob'].mean(), color='darkblue', linestyle='--',
                       label=f'Mean: {false_negatives["break_prob"].mean():.3f}')
        plt.xlabel('Break Probability')
        plt.ylabel('Count')
        plt.title('False Negatives Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

# =========================
# Hàm chính
# =========================
def main():
    print("=== BẮT ĐẦU TESTING MÔ HÌNH TÁCH CÂU ===\n")
    
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load dữ liệu test
    print("[1] Đang load dữ liệu test...")
    dataset = load_test_data(TEST_FILE)
    if dataset is None:
        print("✗ Không thể load dữ liệu test")
        return
    
    test_dataset = dataset['test']
    
    # 2. Load model và tokenizer
    print("[2] Đang load model và tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
        print("✓ Load model thành công")
    except Exception as e:
        print(f"✗ Lỗi khi load model: {e}")
        return
    
    # 3. Load threshold
    print("[3] Đang load ngưỡng tối ưu...")
    try:
        with open(os.path.join(MODEL_DIR, "best_threshold.txt"), "r") as f:
            threshold = float(f.read().strip())
        print(f"✓ Sử dụng ngưỡng: {threshold}")
    except:
        threshold = 0.5
        print(f"⚠️ Không tìm thấy file threshold, sử dụng mặc định: {threshold}")
    
    # 4. Đánh giá mô hình
    print("[4] Đang đánh giá mô hình...")
    true_labels, pred_labels, break_probs, detailed_results = evaluate_model(
        model, tokenizer, test_dataset, threshold
    )
    
    # 5. Tính toán metrics
    print("[5] Đang tính toán độ chính xác...")
    
    accuracy = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average='binary', zero_division=0
    )
    
    # Phân tích theo class
    class_report = classification_report(true_labels, pred_labels, 
                                       target_names=['CONT', 'BREAK'], 
                                       output_dict=True)
    
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH TÁCH CÂU")
    print("="*50)
    print(f"Tổng số mẫu test: {len(test_dataset)}")
    print(f"Tổng số từ: {len(true_labels)}")
    print(f"Số từ BREAK thực tế: {sum(true_labels)}")
    print(f"Số từ BREAK dự đoán: {sum(pred_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\n" + "="*50)
    print("BÁO CÁO CHI TIẾT")
    print("="*50)
    print(classification_report(true_labels, pred_labels, target_names=['CONT', 'BREAK']))
    
    # 6. Lưu kết quả chi tiết
    print("[6] Đang lưu kết quả chi tiết...")
    results_df = pd.DataFrame(detailed_results)
    results_csv_path = os.path.join(OUTPUT_DIR, "detailed_results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Kết quả chi tiết lưu tại: {results_csv_path}")
    
    # 7. Visualization
    print("[7] Đang tạo biểu đồ...")
    
    # Confusion Matrix
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(true_labels, pred_labels, cm_path)
    print(f"✓ Confusion matrix lưu tại: {cm_path}")
    
    # Precision-Recall Curve
    pr_path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
    plot_precision_recall_curve(true_labels, break_probs, pr_path)
    print(f"✓ Precision-Recall curve lưu tại: {pr_path}")
    
    # Threshold Analysis
    thresh_path = os.path.join(OUTPUT_DIR, "threshold_analysis.png")
    best_threshold, best_f1 = plot_threshold_analysis(true_labels, break_probs, thresh_path)
    print(f"✓ Phân tích ngưỡng lưu tại: {thresh_path}")
    print(f"  Ngưỡng tốt nhất: {best_threshold:.3f} (F1: {best_f1:.3f})")
    
    # Error Analysis
    error_path = os.path.join(OUTPUT_DIR, "error_analysis.png")
    plot_error_analysis(results_df, error_path)
    print(f"✓ Phân tích lỗi lưu tại: {error_path}")
    
    # 8. Phân tích lỗi chi tiết
    print("[8] Phân tích lỗi...")
    
    # Thống kê lỗi
    total_predictions = len(results_df)
    correct_predictions = sum(results_df['is_correct'])
    error_predictions = total_predictions - correct_predictions
    
    false_positives = len(results_df[(results_df['true_break'] == 0) & (results_df['pred_break'] == 1)])
    false_negatives = len(results_df[(results_df['true_break'] == 1) & (results_df['pred_break'] == 0)])
    
    print(f"\nPHÂN TÍCH LỖI:")
    print(f"Tổng số dự đoán: {total_predictions}")
    print(f"Số dự đoán đúng: {correct_predictions} ({correct_predictions/total_predictions:.2%})")
    print(f"Số dự đoán sai: {error_predictions} ({error_predictions/total_predictions:.2%})")
    print(f"False Positives (ngắt sai): {false_positives}")
    print(f"False Negatives (bỏ sót ngắt): {false_negatives}")
    
    # Hiển thị một số ví dụ lỗi
    print(f"\nVÍ DỤ LỖI (5 mẫu đầu):")
    error_examples = results_df[results_df['is_correct'] == False].head(5)
    
    for _, row in error_examples.iterrows():
        error_type = "False Positive" if row['true_break'] == 0 else "False Negative"
        print(f"\n{error_type}:")
        print(f"  Từ: '{row['word']}'")
        print(f"  Xác suất BREAK: {row['break_prob']:.3f}")
        print(f"  Văn bản: {row['text'][:100]}...")
    
    # 9. Demo trên một số câu mẫu
    print(f"\n[9] Demo tách câu...")
    demo_texts = [
        "lan có 3 quả táo, mẹ cho thêm 4 quả táo nữa. Lan có tất cả bao nhiêu quả táo",
        "Tổng của 5 và 3 là bao nhiêu? Sau đó nhân với 2 ta được kết quả gì",
        "An có 10 viên bi. Bình có ít hơn An 3 viên bi. Hỏi cả hai bạn có bao nhiêu viên bi?"
    ]
    
    for i, text in enumerate(demo_texts, 1):
        print(f"\n--- Demo {i} ---")
        print(f"Input: {text}")
        
        word_predictions = predict_sentence_breaks(model, tokenizer, text, threshold)
        sentences = []
        current_sentence = []
        
        for pred in word_predictions:
            current_sentence.append(pred['word'])
            if pred['prediction'] == 1:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
        
        if current_sentence:
            sentences.append(' '.join(current_sentence))
        
        print("Kết quả tách câu:")
        for j, sent in enumerate(sentences, 1):
            print(f"  Câu {j}: {sent}")
    
    # 10. Lưu báo cáo tổng hợp
    summary_report = {
        'total_samples': len(test_dataset),
        'total_words': len(true_labels),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'error_analysis': {
            'total_errors': error_predictions,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'error_rate': error_predictions / total_predictions
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "summary_report.json"), 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n" + "="*50)
    print("HOÀN TẤT TESTING")
    print("="*50)
    print(f"✓ Tất cả kết quả được lưu trong: {OUTPUT_DIR}")
    print(f"✓ Độ chính xác tổng thể: {accuracy:.2%}")
    print(f"✓ F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()