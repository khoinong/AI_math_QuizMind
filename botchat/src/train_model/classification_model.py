import os
import json
import random
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf  

# ⚠️ Import cụ thể TFRobertaModel thay vì TFAutoModel để tránh lỗi mapping ở một số bản transformers
try:
    from transformers import AutoTokenizer, TFRobertaModel
except Exception as e:
    # Gợi ý rõ ràng nếu môi trường thiếu lớp TF cho Roberta
    raise ImportError(
        "Không thể import AutoTokenizer/TFRobertaModel từ transformers.\n"
        "Hãy kiểm tra phiên bản: pip show transformers ; nên dùng transformers >= 4.38\n"
        "Nếu bạn dùng Python 3.13 trên Windows, tokenizers có thể cần Rust để build.\n"
        "Khuyến nghị: Python 3.10/3.11 + 'pip install \"transformers>=4.38\"'.\n"
    ) from e

from sklearn.metrics import classification_report, f1_score

# =========================
# Cấu hình chung
# =========================
MODEL_NAME = "vinai/phobert-base"   # PhoBERT base (Roberta)
MAX_LEN = 96                         # Độ dài tối đa chuỗi mã hoá (Trade-off: dài hơn -> tốn VRAM)
BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5                           # LR nhỏ cho fine-tuning Transformer
PATIENCE = 3                        # EarlyStopping kiềm overfitting
ARTIFACT_DIR = "src/model/classification_model"        # Nơi lưu model/metadata
SEED = 42

# Dữ liệu: dùng demo mặc định; chuyển sang CSV thật bằng cách bật cờ bên dưới
DATASET_CUSTOM = True              # True để đọc CSV thật
CSV_PATH = "data/math_problem_label.csv"              # CSV cần cột: text, labels (ví dụ: "news;business")
LABEL_NAMES = ["basic","basic_word","ownership","ratio","comparison"]  # chỉnh theo bộ nhãn thực tế
VAL_SPLIT = 0.2                     # Tỉ lệ validation khi không có split sẵn

# =========================
# Tiện ích reproducibility & I/O
# =========================
def set_seed(seed: int = 42) -> None:
    """Cố định seed cho random/NumPy/TensorFlow để chạy lặp lại có kết quả tương tự."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: str) -> None:
    """Tạo thư mục nếu chưa có (lưu artifacts)."""
    os.makedirs(path, exist_ok=True)


def to_one_hot(label_str: str, label2id: Dict[str, int]) -> np.ndarray:
    hot = np.zeros(len(label2id), dtype="float32")
    if isinstance(label_str, str) and label_str.strip():
        first_label = label_str.split(";")[0].strip()
        if first_label in label2id:
            hot[label2id[first_label]] = 1.0
    return hot


# =========================
# Tải backbone & tokenizer PhoBERT
# =========================
def build_backbone_and_tokenizer(model_name: str):
    """Load tokenizer và backbone TFRobertaModel.
    PhoBERT có checkpoint PyTorch, nên ta dùng from_pt=True để auto-convert sang TF.
    """
    # use_fast=False: PhoBERT thường dùng tokenizer kiểu Roberta cũ; fast có thể lệch behavior ở edge-case
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    try:
        backbone = TFRobertaModel.from_pretrained(model_name, from_pt=True)
    except Exception as e:
        # Trường hợp transformers thiếu lớp TF cho Roberta (hoặc tokenizers không tương thích)
        raise RuntimeError(
            "Không load được TFRobertaModel.\n"
            "• Kiểm tra: tensorflow >= 2.12, transformers >= 4.38\n"
            "• Nếu bạn đang ở Python 3.13 trên Windows, hãy cài Rust (cargo) hoặc dùng Python 3.10/3.11.\n"
            f"Chi tiết gốc: {e}"
        )
    return tokenizer, backbone

# =========================
# Mã hoá văn bản & tf.data
# =========================
def encode_texts(tokenizer, texts: List[str], max_len: int) -> Dict[str, tf.Tensor]:
    """Tokenize danh sách văn bản thành input_ids & attention_mask (tensors TF)."""
    enc = tokenizer(
        texts,
        padding=True,           # pad theo batch -> shape đồng nhất
        truncation=True,        # cắt bớt nếu dài hơn max_len
        max_length=max_len,
        return_tensors="tf",
    )
    # Trả về dict phù hợp input Keras Functional API
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


def make_tf_dataset(features: Dict[str, tf.Tensor], labels: np.ndarray, batch_size: int, shuffle: bool = False):
    """Đóng gói (features, labels) thành tf.data.Dataset để train hiệu quả hơn."""
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        # buffer_size = len(labels): shuffle đủ lớn để trộn ngẫu nhiên tốt ở dataset nhỏ
        ds = ds.shuffle(buffer_size=len(labels))
    ds = ds.batch(batch_size)
    # (tuỳ chọn) .prefetch(tf.data.AUTOTUNE) để overlapping I/O & compute
    return ds.prefetch(tf.data.AUTOTUNE)

# =========================
# Xây kiến trúc Keras
# =========================
def build_classifier(backbone: tf.keras.Model, num_classes: int) -> tf.keras.Model:
    input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    last_hidden = backbone(input_ids, attention_mask=attention_mask)[0]
    cls = last_hidden[:, 0, :]  # (B, H)

    x = tf.keras.layers.Dense(256, activation="relu")(cls)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Softmax: chỉ ra phân phối xác suất giữa các nhãn
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)

    # categorical_crossentropy cho single-label
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =========================
# Tối ưu ngưỡng dựa trên F1 từng nhãn
# =========================
# Vì sigmoid trả về xác suất độc lập, ngưỡng 0.5 chỉ là mặc định. Tối ưu threshold per-label thường cải F1 đáng kể.

def grid_search_thresholds(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str], grid=None) -> Tuple[List[float], Dict[str, float]]:
    """Quét ngưỡng theo từng nhãn để tối đa F1 (mặc định micro cho từng nhãn)."""
    if grid is None:
        # Lưới ngưỡng: 0.20 .. 0.80 (bước 0.05)
        grid = np.linspace(0.2, 0.8, 13)

    best_thresholds: List[float] = []
    per_label_best_f1: Dict[str, float] = {}

    for j, name in enumerate(label_names):
        best_f1 = -1.0
        best_t = 0.5
        y_true_j = y_true[:, j]
        y_prob_j = y_prob[:, j]
        for t in grid:
            y_pred_j = (y_prob_j >= t).astype(int)
            f1 = f1_score(y_true_j, y_pred_j, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best_thresholds.append(best_t)
        per_label_best_f1[name] = best_f1

    return best_thresholds, per_label_best_f1


def apply_thresholds(y_prob: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """Áp dụng ngưỡng theo từng cột (nhãn)."""
    thr = np.array(thresholds).reshape(1, -1)
    return (y_prob >= thr).astype(int)

# =========================
# Dataset demo & CSV
# =========================
def load_csv_dataset(csv_path: str, label_names: List[str]) -> Tuple[List[str], np.ndarray]:
    """Đọc dữ liệu thật từ CSV có cột text, labels (chuỗi nhãn phân tách bằng ';')."""
    df = pd.read_csv(csv_path)
    label2id = {l: i for i, l in enumerate(label_names)}
    texts = df["text"].astype(str).tolist()
    y = np.vstack([to_one_hot(s, label2id) for s in df["labels"]])
    return texts, y

# =========================
# Main
# =========================
if __name__ == "__main__":
    set_seed(SEED)
    ensure_dir(ARTIFACT_DIR)

    # ----- 1) Load dữ liệu -----
    if DATASET_CUSTOM:
        # Đọc CSV thật
        texts, labels = load_csv_dataset(CSV_PATH, LABEL_NAMES)
        # Tạo split train/val ngẫu nhiên nếu CSV chưa có split sẵn
        idx = np.arange(len(texts))
        np.random.shuffle(idx)
        split = int((1.0 - VAL_SPLIT) * len(texts))
        train_idx, val_idx = idx[:split], idx[split:]
    else:
        exit("Chưa hỗ trợ dataset mặc định, vui lòng bật DATASET_CUSTOM=True và cung cấp CSV hợp lệ.")  

    texts_train = [texts[i] for i in train_idx]
    y_train = labels[train_idx]
    texts_val = [texts[i] for i in val_idx]
    y_val = labels[val_idx]

    num_classes = len(LABEL_NAMES)

    # ----- 2) Tokenizer & Backbone PhoBERT -----
    print("\n[Info] Loading PhoBERT backbone & tokenizer ...")
    tokenizer, backbone = build_backbone_and_tokenizer(MODEL_NAME)

    # ----- 3) Mã hoá văn bản -----
    print("[Info] Tokenizing ...")
    train_enc = encode_texts(tokenizer, texts_train, MAX_LEN)
    val_enc = encode_texts(tokenizer, texts_val, MAX_LEN)

    # tf.data để train hiệu quả hơn (prefetch giúp overlap I/O)
    train_ds = make_tf_dataset(train_enc, y_train, BATCH_SIZE, shuffle=True)
    val_ds = make_tf_dataset(val_enc, y_val, BATCH_SIZE, shuffle=False)

    # ----- 4) Xây model Keras -----
    print("[Info] Building classifier ...")
    model = build_classifier(backbone, num_classes)
    model.summary()

    # ----- 5) Huấn luyện -----
    print("[Info] Training ...")
    callbacks = [
        # EarlyStopping dựa trên AUC (multi_label) — phù hợp khi class imbalance
        tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_auc", mode="max"),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # ----- 6) Đánh giá + Tối ưu ngưỡng -----
    print("[Info] Evaluating on validation set ...")
    val_probs = model.predict(val_ds)

    # Báo cáo với ngưỡng mặc định 0.5 để tham chiếu
    val_pred_default = (val_probs >= 0.5).astype(int)
    print("\n== Classification report @threshold=0.5 ==")
    print(classification_report(y_val, val_pred_default, target_names=LABEL_NAMES, zero_division=0))

    # Tìm ngưỡng tốt nhất cho từng nhãn để đẩy F1
    print("[Info] Tuning per-label thresholds ...")
    best_thresholds, per_label_f1 = grid_search_thresholds(y_val, val_probs, LABEL_NAMES)
    val_pred_tuned = apply_thresholds(val_probs, best_thresholds)

    micro_f1 = f1_score(y_val, val_pred_tuned, average="micro", zero_division=0)
    macro_f1 = f1_score(y_val, val_pred_tuned, average="macro", zero_division=0)

    print("\n== Tuned thresholds (per label) ==")
    print({name: thr for name, thr in zip(LABEL_NAMES, best_thresholds)})
    print("Per-label best F1:")
    print(per_label_f1)
    print(f"Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f}")

    # ----- 7) Demo dự đoán -----
    test_texts = [
        "1 quả táo + 1 quả táo = bao nhiêu quả táo?",
        "2 quả cam + 3 quả cam = bao nhiêu quả cam?",
        "1 quả táo + 2 quả cam = bao nhiêu quả táo và cam?",
        " 1 + 5 = ?",
        "2 * 3 = ?",
        "3 / 4 = ?",
        "5 - 2 = ?",
        " 4 + 6 + 2",      
    ]
    test_enc = encode_texts(tokenizer, test_texts, MAX_LEN)
    test_probs = model.predict(test_enc)
    test_pred = np.argmax(test_probs, axis=1)  # lấy chỉ số nhãn lớn nhất
    def decode_label(idx: int) -> str:
        return LABEL_NAMES[idx]

    for t, idx, prob in zip(test_texts, test_pred, test_probs):
        print("\nVăn bản:", t)
        print("Nhãn dự đoán:", decode_label(idx))
        print("Xác suất:", {LABEL_NAMES[i]: float(f"{p:.3f}") for i, p in enumerate(prob)})

    # ----- 8) Lưu artifacts để deploy/inference ngoài script -----
    print("\n[Info] Saving artifacts ...")
    ensure_dir(ARTIFACT_DIR)

    # Lưu SavedModel: gồm kiến trúc + trọng số + signature — thuận tiện cho TF Serving
    saved_path = os.path.join(ARTIFACT_DIR, "saved_model")
    model.save(saved_path)

    # Lưu tên nhãn & thresholds để tái sử dụng lúc inference/serving
    with open(os.path.join(ARTIFACT_DIR, "label_names.json"), "w", encoding="utf-8") as f:
        json.dump(LABEL_NAMES, f, ensure_ascii=False, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "thresholds.json"), "w", encoding="utf-8") as f:
        json.dump({name: thr for name, thr in zip(LABEL_NAMES, best_thresholds)}, f, ensure_ascii=False, indent=2)

    print(f"[Done] Artifacts saved to: {ARTIFACT_DIR}")
