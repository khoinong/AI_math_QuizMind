import os
import json
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import tensorflow as tf  
import matplotlib.pyplot as plt

# Import transformers
try:
    from transformers import AutoTokenizer, TFRobertaModel
except Exception as e:
    raise ImportError(
        "Không thể import AutoTokenizer/TFRobertaModel từ transformers.\n"
        "Hãy kiểm tra phiên bản: pip show transformers ; nên dùng transformers >= 4.38\n"
        "Khuyến nghị: Python 3.10/3.11 + 'pip install \"transformers>=4.38\"'.\n"
    ) from e

from sklearn.metrics import classification_report, f1_score


# =========================
# Configuration Class
# =========================
@dataclass
class ClassificationConfig:
    """Cấu hình cho mô hình phân loại"""
    model_name: str = "vinai/phobert-base"
    max_len: int = 96
    batch_size: int = 8
    epochs: int = 10
    learning_rate: float = 2e-5
    patience: int = 3
    artifact_dir: str = "src/model/classification_model"
    seed: int = 42
    dataset_custom: bool = True
    csv_path: str = "data/math_problem_label.csv"
    label_names: List[str] = None
    val_split: float = 0.2

    def __post_init__(self):
        if self.label_names is None:
            self.label_names = ["basic", "basic_word", "ownership", "ratio", "comparison"]


# =========================
# Utility Classes
# =========================
class EnvUtils:
    """Tiện ích môi trường và seed"""
    
    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """Cố định seed cho reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    @staticmethod
    def ensure_dir(path: str) -> None:
        """Tạo thư mục nếu chưa có"""
        os.makedirs(path, exist_ok=True)


# =========================
# Data Processing Classes
# =========================
class TextEncoder:
    """Xử lý encoding văn bản"""
    
    def __init__(self, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def encode_texts(self, texts: List[str]) -> Dict[str, tf.Tensor]:
        """Tokenize danh sách văn bản"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="tf",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class DatasetBuilder:
    """Xây dựng tf.data.Dataset"""
    
    @staticmethod
    def create_dataset(features: Dict[str, tf.Tensor], labels: np.ndarray, 
                      batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
        """Tạo tf.data.Dataset từ features và labels"""
        ds = tf.data.Dataset.from_tensor_slices((features, labels))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(labels))
        ds = ds.batch(batch_size)
        return ds.prefetch(tf.data.AUTOTUNE)


class DataLoader:
    """Tải dữ liệu từ CSV"""
    
    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.label2id = {l: i for i, l in enumerate(label_names)}

    def _to_one_hot(self, label_str: str) -> np.ndarray:
        """Chuyển chuỗi nhãn thành one-hot encoding"""
        hot = np.zeros(len(self.label2id), dtype="float32")
        if isinstance(label_str, str) and label_str.strip():
            first_label = label_str.split(";")[0].strip()
            if first_label in self.label2id:
                hot[self.label2id[first_label]] = 1.0
        return hot

    def load_csv(self, csv_path: str) -> Tuple[List[str], np.ndarray]:
        """Đọc dữ liệu từ CSV"""
        df = pd.read_csv(csv_path)
        texts = df["text"].astype(str).tolist()
        y = np.vstack([self._to_one_hot(s) for s in df["labels"]])
        return texts, y

    def create_train_val_split(self, texts: List[str], labels: np.ndarray, 
                              val_split: float) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """Chia dữ liệu thành train và validation"""
        idx = np.arange(len(texts))
        np.random.shuffle(idx)
        split = int((1.0 - val_split) * len(texts))
        train_idx, val_idx = idx[:split], idx[split:]
        
        texts_train = [texts[i] for i in train_idx]
        texts_val = [texts[i] for i in val_idx]
        y_train = labels[train_idx]
        y_val = labels[val_idx]
        
        return texts_train, texts_val, y_train, y_val


# =========================
# Model Building Classes
# =========================
class ModelBuilder:
    """Xây dựng mô hình Keras"""
    
    @staticmethod
    def build_backbone_and_tokenizer(model_name: str):
        """Load tokenizer và backbone TFRobertaModel"""
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        
        try:
            backbone = TFRobertaModel.from_pretrained(model_name, from_pt=True)
        except Exception as e:
            raise RuntimeError(
                "Không load được TFRobertaModel.\n"
                "• Kiểm tra: tensorflow >= 2.12, transformers >= 4.38\n"
                f"Chi tiết gốc: {e}"
            )
        return tokenizer, backbone

    @staticmethod
    def build_classifier(backbone: tf.keras.Model, num_classes: int, learning_rate: float) -> tf.keras.Model:
        """Xây dựng classifier model"""
        input_ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="input_ids")
        attention_mask = tf.keras.Input(shape=(None,), dtype=tf.int32, name="attention_mask")

        last_hidden = backbone(input_ids, attention_mask=attention_mask)[0]
        cls = last_hidden[:, 0, :]

        x = tf.keras.layers.Dense(256, activation="relu")(cls)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model


# =========================
# Evaluation Classes
# =========================
class ThresholdOptimizer:
    """Tối ưu ngưỡng cho từng nhãn"""
    
    @staticmethod
    def grid_search(y_true: np.ndarray, y_prob: np.ndarray, label_names: List[str], 
                   grid: np.ndarray = None) -> Tuple[List[float], Dict[str, float]]:
        """Quét ngưỡng để tối đa F1"""
        if grid is None:
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

    @staticmethod
    def apply_thresholds(y_prob: np.ndarray, thresholds: List[float]) -> np.ndarray:
        """Áp dụng ngưỡng cho từng nhãn"""
        thr = np.array(thresholds).reshape(1, -1)
        return (y_prob >= thr).astype(int)


class Evaluator:
    """Đánh giá mô hình"""
    
    def __init__(self, label_names: List[str]):
        self.label_names = label_names

    def evaluate(self, model, val_ds, y_val) -> Dict:
        """Đánh giá mô hình trên tập validation"""
        val_probs = model.predict(val_ds)
        
        # Báo cáo với ngưỡng mặc định
        val_pred_default = (val_probs >= 0.5).astype(int)
        
        return {
            "predictions": val_probs,
            "pred_default": val_pred_default,
            "y_true": y_val,
            "report_default": classification_report(y_val, val_pred_default, 
                                                   target_names=self.label_names, 
                                                   zero_division=0)
        }


# =========================
# Visualization Classes
# =========================
class Visualizer:
    """Vẽ biểu đồ đánh giá"""
    
    def __init__(self, artifact_dir: str):
        self.plot_dir = os.path.join(artifact_dir, "plots")
        EnvUtils.ensure_dir(self.plot_dir)

    def plot_loss_curve(self, history):
        """Vẽ biểu đồ loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(history.history["loss"], label="train_loss")
        plt.plot(history.history["val_loss"], label="val_loss")
        plt.title("Training - Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "loss_curve.png"))
        plt.close()
        print(f"[Visualization] Saved loss curve to: {os.path.join(self.plot_dir, 'loss_curve.png')}")

    def plot_accuracy_curve(self, history):
        """Vẽ biểu đồ accuracy"""
        if "accuracy" in history.history:
            plt.figure(figsize=(8, 5))
            plt.plot(history.history["accuracy"], label="train_acc")
            plt.plot(history.history["val_accuracy"], label="val_acc")
            plt.title("Training - Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, "accuracy_curve.png"))
            plt.close()
            print(f"[Visualization] Saved accuracy curve to: {os.path.join(self.plot_dir, 'accuracy_curve.png')}")

    def plot_f1_scores(self, per_label_f1: Dict[str, float]):
        """Vẽ biểu đồ F1 theo từng nhãn"""
        plt.figure(figsize=(8, 5))
        names = list(per_label_f1.keys())
        values = list(per_label_f1.values())
        plt.bar(names, values)
        plt.ylim(0, 1)
        plt.title("Per-label F1 Score After Threshold Tuning")
        plt.xlabel("Label")
        plt.ylabel("F1 Score")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "f1_threshold_tuning.png"))
        plt.close()
        print(f"[Visualization] Saved F1 chart to: {os.path.join(self.plot_dir, 'f1_threshold_tuning.png')}")


# =========================
# Model Persistence Classes
# =========================
class ModelArtifactManager:
    """Quản lý lưu và tải artifacts (mô hình, tokenizer, metadata)"""
    
    def __init__(self, artifact_dir: str):
        self.artifact_dir = artifact_dir
        EnvUtils.ensure_dir(artifact_dir)

    def save_model(self, model: tf.keras.Model):
        """Lưu mô hình"""
        saved_path = os.path.join(self.artifact_dir, "saved_model")
        model.save(saved_path)
        print(f"[Info] Saved model to: {saved_path}")

    def save_tokenizer(self, tokenizer):
        """Lưu tokenizer"""
        tokenizer.save_pretrained(self.artifact_dir)
        print(f"[Info] Saved tokenizer to: {self.artifact_dir}")

    def save_metadata(self, label_names: List[str], thresholds: Dict[str, float]):
        """Lưu metadata (nhãn và ngưỡng)"""
        with open(os.path.join(self.artifact_dir, "label_names.json"), "w", encoding="utf-8") as f:
            json.dump(label_names, f, ensure_ascii=False, indent=2)

        with open(os.path.join(self.artifact_dir, "thresholds.json"), "w", encoding="utf-8") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)

        print(f"[Info] Saved metadata to: {self.artifact_dir}")


# =========================
# Main Training Pipeline
# =========================
class ClassificationPipeline:
    """Pipeline huấn luyện toàn bộ mô hình phân loại"""
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.tokenizer = None
        self.backbone = None
        self.model = None
        self.history = None
        self.eval_results = None
        
        # Initialize components
        self.text_encoder = None
        self.data_loader = DataLoader(config.label_names)
        self.model_builder = ModelBuilder()
        self.evaluator = Evaluator(config.label_names)
        self.visualizer = Visualizer(config.artifact_dir)
        self.optimizer = ThresholdOptimizer()
        self.artifact_manager = ModelArtifactManager(config.artifact_dir)

    def setup(self):
        """Khởi tạo seed và tải backbone"""
        EnvUtils.set_seed(self.config.seed)
        print("[Info] Loading PhoBERT backbone & tokenizer...")
        self.tokenizer, self.backbone = self.model_builder.build_backbone_and_tokenizer(
            self.config.model_name
        )
        self.text_encoder = TextEncoder(self.tokenizer, self.config.max_len)

    def load_data(self) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """Tải dữ liệu từ CSV"""
        print("[Info] Loading data from CSV...")
        texts, labels = self.data_loader.load_csv(self.config.csv_path)
        
        # Chia train/val
        texts_train, texts_val, y_train, y_val = self.data_loader.create_train_val_split(
            texts, labels, self.config.val_split
        )
        
        print(f"[Info] Train samples: {len(texts_train)}, Val samples: {len(texts_val)}")
        return texts_train, texts_val, y_train, y_val

    def prepare_datasets(self, texts_train: List[str], texts_val: List[str], 
                        y_train: np.ndarray, y_val: np.ndarray):
        """Chuẩn bị tf.data datasets"""
        print("[Info] Tokenizing...")
        train_enc = self.text_encoder.encode_texts(texts_train)
        val_enc = self.text_encoder.encode_texts(texts_val)
        
        train_ds = DatasetBuilder.create_dataset(train_enc, y_train, self.config.batch_size, shuffle=True)
        val_ds = DatasetBuilder.create_dataset(val_enc, y_val, self.config.batch_size, shuffle=False)
        
        return train_ds, val_ds

    def build_model(self):
        """Xây dựng mô hình"""
        print("[Info] Building classifier...")
        self.model = self.model_builder.build_classifier(
            self.backbone, 
            len(self.config.label_names),
            self.config.learning_rate
        )
        self.model.summary()

    def train(self, train_ds, val_ds):
        """Huấn luyện mô hình"""
        print("[Info] Training...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=self.config.patience, 
                restore_best_weights=True, 
                monitor="val_accuracy", 
                mode="max"
            ),
        ]
        
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1,
        )

    def evaluate(self, val_ds, y_val):
        """Đánh giá và tối ưu ngưỡng"""
        print("[Info] Evaluating on validation set...")
        self.eval_results = self.evaluator.evaluate(self.model, val_ds, y_val)
        
        print("\n== Classification report @threshold=0.5 ==")
        print(self.eval_results["report_default"])
        
        # Tối ưu ngưỡng
        print("[Info] Tuning per-label thresholds...")
        best_thresholds, per_label_f1 = self.optimizer.grid_search(
            y_val, self.eval_results["predictions"], self.config.label_names
        )
        
        val_pred_tuned = self.optimizer.apply_thresholds(self.eval_results["predictions"], best_thresholds)
        micro_f1 = f1_score(y_val, val_pred_tuned, average="micro", zero_division=0)
        macro_f1 = f1_score(y_val, val_pred_tuned, average="macro", zero_division=0)
        
        print("\n== Tuned thresholds (per label) ==")
        print({name: thr for name, thr in zip(self.config.label_names, best_thresholds)})
        print("Per-label best F1:")
        print(per_label_f1)
        print(f"Micro-F1: {micro_f1:.4f} | Macro-F1: {macro_f1:.4f}")
        
        return best_thresholds, per_label_f1

    def visualize(self, per_label_f1: Dict[str, float]):
        """Vẽ biểu đồ"""
        print("[Info] Saving visualizations...")
        self.visualizer.plot_loss_curve(self.history)
        self.visualizer.plot_accuracy_curve(self.history)
        self.visualizer.plot_f1_scores(per_label_f1)

    def save_artifacts(self, best_thresholds: List[float]):
        """Lưu mô hình và metadata"""
        print("[Info] Saving artifacts...")
        self.artifact_manager.save_model(self.model)
        self.artifact_manager.save_tokenizer(self.tokenizer)
        
        threshold_dict = {name: thr for name, thr in zip(self.config.label_names, best_thresholds)}
        self.artifact_manager.save_metadata(self.config.label_names, threshold_dict)

    def predict_demo(self):
        """Demo dự đoán"""
        test_texts = [
            "1 quả táo + 1 quả táo = bao nhiêu quả táo?",
            "2 quả cam + 3 quả cam = bao nhiêu quả cam?",
            "1 quả táo + 2 quả cam = bao nhiêu quả táo và cam?",
            "1 + 5 = ?",
            "2 * 3 = ?",
        ]
        
        print("\n[Demo] Predicting on test texts...")
        test_enc = self.text_encoder.encode_texts(test_texts)
        test_probs = self.model.predict(test_enc)
        test_pred = np.argmax(test_probs, axis=1)
        
        for t, idx, prob in zip(test_texts, test_pred, test_probs):
            print(f"\nVăn bản: {t}")
            print(f"Nhãn dự đoán: {self.config.label_names[idx]}")
            print(f"Xác suất: {{{', '.join([f'{name}: {p:.3f}' for name, p in zip(self.config.label_names, prob)])}}}")

    def run(self):
        """Chạy toàn bộ pipeline"""
        print("\n" + "="*50)
        print("CLASSIFICATION MODEL TRAINING PIPELINE")
        print("="*50)
        
        self.setup()
        texts_train, texts_val, y_train, y_val = self.load_data()
        train_ds, val_ds = self.prepare_datasets(texts_train, texts_val, y_train, y_val)
        self.build_model()
        self.train(train_ds, val_ds)
        best_thresholds, per_label_f1 = self.evaluate(val_ds, y_val)
        self.visualize(per_label_f1)
        self.save_artifacts(best_thresholds)
        self.predict_demo()
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)


# =========================
# Main Entry Point
# =========================
if __name__ == "__main__":
    config = ClassificationConfig(
        model_name="vinai/phobert-base",
        artifact_dir="src/model/classification_model",
        batch_size=8,
        epochs=10,
    )
    
    pipeline = ClassificationPipeline(config)
    pipeline.run()
