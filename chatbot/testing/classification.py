import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# =========================
# Cấu hình
# =========================
MODEL_DIR = "src/model/classification_model"
TEST_CSV_PATH = "data/test_classification.csv"
MAX_LEN = 96
BATCH_SIZE = 16
NUM_SAMPLES = 100

# =========================
# Hàm tiện ích TensorFlow
# =========================
def load_classification_model(model_dir):
    """Load model, tokenizer và label names từ classification_model"""
    # Load label names
    with open(os.path.join(model_dir, "label_names.json"), "r", encoding="utf-8") as f:
        label_names = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    
    # Load TensorFlow SavedModel
    model_path = os.path.join(model_dir, "saved_model")
    model = tf.saved_model.load(model_path)
    
    return model, tokenizer, label_names

def load_test_data(csv_path, label_names, num_samples=100):
    """Load dữ liệu test từ CSV"""
    df = pd.read_csv(csv_path)
    
    # Fix: Remove trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    if len(df) > num_samples:
        df_test = df.sample(n=num_samples, random_state=42)
    else:
        df_test = df
        print(f"[Warning] Dataset chỉ có {len(df)} mẫu, sẽ test toàn bộ")
    
    texts = df_test["text"].astype(str).tolist()
    
    # Chuyển đổi labels
    label2id = {label: i for i, label in enumerate(label_names)}
    
    def to_label_id(label_str):
        if isinstance(label_str, str) and label_str.strip():
            first_label = label_str.split(";")[0].strip()
            return label2id.get(first_label, 0)
        return 0
    
    labels = [to_label_id(s) for s in df_test["labels"]]
    
    return texts, labels, df_test

def encode_texts_tf(tokenizer, texts, max_len):
    """Mã hóa văn bản cho TensorFlow"""
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="tf",
    )
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    """Vẽ confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =========================
# Main Testing Script
# =========================
def main():
    print("=== BẮT ĐẦU TESTING MÔ HÌNH PHÂN LOẠI ===\n")
    
    # 1. Load model và artifacts
    print("[1] Đang load model và artifacts...")
    try:
        model, tokenizer, label_names = load_classification_model(MODEL_DIR)
        print(f"✓ Loaded {len(label_names)} classes: {label_names}")
    except Exception as e:
        print(f"✗ Lỗi khi load model: {e}")
        return
    
    # 2. Load dữ liệu test
    print(f"\n[2] Đang load dữ liệu test...")
    try:
        test_texts, true_labels, df_test = load_test_data(TEST_CSV_PATH, label_names, NUM_SAMPLES)
        print(f"✓ Loaded {len(test_texts)} mẫu test")
    except Exception as e:
        print(f"✗ Lỗi khi load dữ liệu test: {e}")
        return
    
    # 3. Dự đoán
    print(f"\n[3] Đang dự đoán...")
    try:
        test_enc = encode_texts_tf(tokenizer, test_texts, MAX_LEN)
        
        # Call SavedModel with the correct signature
        infer = model.signatures["serving_default"]
        output = infer(input_ids=test_enc["input_ids"], attention_mask=test_enc["attention_mask"])
        
        # Get predictions from output (output name is 'dense_1')
        test_probs = output["dense_1"].numpy() if hasattr(output["dense_1"], "numpy") else output["dense_1"]
        predicted_labels = np.argmax(test_probs, axis=1)
        print(f"✓ Dự đoán thành công cho {len(test_texts)} mẫu")
    except Exception as e:
        print(f"✗ Lỗi khi dự đoán: {e}")
        return
    
    # 4. Tính toán metrics
    print(f"\n[4] Đang tính toán độ chính xác...")
    
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
    
    print(f"\n=== KẾT QUẢ TỔNG QUAN ===")
    print(f"Tổng số mẫu test: {len(test_texts)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-Score (macro): {f1:.4f}")
    
    # 5. Báo cáo chi tiết
    print(f"\n=== BÁO CÁO CHI TIẾT ===")
    print(classification_report(true_labels, predicted_labels, target_names=label_names, zero_division=0))
    
    # 6. Lưu kết quả
    print(f"\n[5] Đang lưu kết quả...")
    results_dir = "test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Tạo DataFrame kết quả
    results_df = pd.DataFrame({
        'Văn bản': test_texts,
        'Nhãn thực tế': [label_names[i] for i in true_labels],
        'Nhãn dự đoán': [label_names[i] for i in predicted_labels],
        'Đúng/Sai': [true == pred for true, pred in zip(true_labels, predicted_labels)]
    })
    
    results_csv_path = os.path.join(results_dir, "classification_test_results.csv")
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Kết quả lưu tại: {results_csv_path}")
    
    # 7. Visualization
    cm_path = os.path.join(results_dir, "classification_confusion_matrix.png")
    plot_confusion_matrix(true_labels, predicted_labels, label_names, cm_path)
    print(f"✓ Confusion matrix lưu tại: {cm_path}")
    
    # 8. Phân tích lỗi
    wrong_predictions = results_df[results_df['Đúng/Sai'] == False]
    
    if len(wrong_predictions) > 0:
        print(f"\n[6] Phân tích {len(wrong_predictions)} mẫu bị dự đoán sai:")
        for _, row in wrong_predictions.head(10).iterrows():  # Hiển thị 10 mẫu đầu
            print(f"\nVăn bản: {row['Văn bản'][:80]}...")
            print(f"Thực tế: {row['Nhãn thực tế']} | Dự đoán: {row['Nhãn dự đoán']}")
    else:
        print("✓ Tuyệt vời! Không có mẫu nào bị dự đoán sai.")
    
    print(f"\n=== HOÀN TẤT TESTING ===")
    print(f"Kết quả được lưu trong thư mục: {results_dir}")

if __name__ == "__main__":
    main()