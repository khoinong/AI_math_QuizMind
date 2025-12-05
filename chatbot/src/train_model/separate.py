# -*- coding: utf-8 -*-
"""Huấn luyện mô hình tách câu - Phiên bản hướng đối tượng"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    DataCollatorForTokenClassification, 
    TrainingArguments, 
    Trainer,
    TrainerCallback,
    RobertaTokenizerFast
)
from datasets import DatasetDict, Dataset
from tqdm.auto import tqdm
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, f1_score
import json
import os


@dataclass
class ModelConfig:
    """Cấu hình cho mô hình tách câu"""
    model_name: str = "vinai/phobert-base-v2"
    labels: List[str] = field(default_factory=lambda: ["CONT", "BREAK"])
    max_length: int = 512
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    output_dir: str = "src/model/sent_split_model"
    logging_dir: str = "output/logs"
    save_total_limit: int = 2
    max_grad_norm: float = 1.0


class MetricsLoggerCallback(TrainerCallback):
    """Callback để ghi lại các metrics trong quá trình huấn luyện"""
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.eval_f1 = []
        self.eval_precision = []
        self.eval_recall = []
        self.eval_accuracy = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs and 'epoch' in logs:
                self.train_loss.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_loss.append(logs['eval_loss'])
            if 'eval_f1' in logs:
                self.eval_f1.append(logs['eval_f1'])
            if 'eval_precision' in logs:
                self.eval_precision.append(logs['eval_precision'])
            if 'eval_recall' in logs:
                self.eval_recall.append(logs['eval_recall'])
            if 'eval_accuracy' in logs:
                self.eval_accuracy.append(logs['eval_accuracy'])


class SentenceSplitterDatasetProcessor:
    """Xử lý dataset cho bài toán tách câu"""
    def __init__(self, tokenizer, config: ModelConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.label2id = {label: i for i, label in enumerate(config.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
    def build_labels_for_example(self, text: str, word_break_indices: Set[int]) -> Dict[str, List[int]]:
        """Xây dựng nhãn cho từng từ trong câu"""
        words = text.split()
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            padding=False,
        )
        
        labels = np.full(len(encoding.input_ids), -100, dtype=int)
        word_ids = encoding.word_ids()
        
        for i in range(len(encoding.input_ids)):
            word_idx = word_ids[i]
            
            if word_idx is None:
                continue
                
            is_last_token = (i == len(encoding.input_ids)-1) or (word_ids[i+1] != word_idx)
            
            if is_last_token:
                label = "BREAK" if word_idx in word_break_indices else "CONT"
                labels[i] = self.label2id[label]
        
        encoding["labels"] = labels.tolist()
        return encoding
    
    def preprocess_batch(self, batch: Dict[str, List]) -> Dict[str, List]:
        """Tiền xử lý một batch dữ liệu"""
        texts = batch["text"]
        breaks = batch["breaks"]
        out = {k: [] for k in ["input_ids", "attention_mask", "labels"]}
        
        for text, break_indices in zip(texts, breaks):
            enc = self.build_labels_for_example(text, set(break_indices))
            for key in out:
                out[key].append(enc[key])
        
        return out
    
    def load_and_preprocess(self, train_path: str, val_path: str) -> DatasetDict:
        """Tải và tiền xử lý dataset"""
        from datasets import load_dataset
        
        print("Đang tải dữ liệu...")
        raw_data = load_dataset("json", data_files={
            "train": train_path,
            "validation": val_path
        })
        
        # Áp dụng tiền xử lý
        processed_data = raw_data.map(
            self.preprocess_batch, 
            batched=True,
            remove_columns=raw_data["train"].column_names
        )  
        return processed_data


class SentenceSplitterModel:
    """Lớp chính cho mô hình tách câu"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.processor = None
        self.metrics_callback = None
        self.label2id = None
        self.id2label = None
        self.best_threshold = 0.5
        
        # Khởi tạo label mappings
        self._init_label_mappings()
        
    def _init_label_mappings(self):
        """Khởi tạo ánh xạ nhãn"""
        self.label2id = {label: i for i, label in enumerate(self.config.labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
    
    def load_tokenizer(self, tokenizer_path: str = None) -> None:
        """Tải hoặc tạo tokenizer"""
        if tokenizer_path and os.path.exists(tokenizer_path):
            print(f"Tải tokenizer từ {tokenizer_path}...")
            self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path)
        else:
            print("Tạo tokenizer mới từ PhoBERT...")
            self.tokenizer = RobertaTokenizerFast.from_pretrained(
                self.config.model_name,
                use_fast=True,
                add_prefix_space=True,
                model_max_length=self.config.max_length
            )
            
            # Lưu tokenizer nếu có đường dẫn
            if tokenizer_path:
                os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
                self.tokenizer.save_pretrained(tokenizer_path)
        
        print(f"Tokenizer type: {type(self.tokenizer)}")
        if not self.tokenizer.is_fast:
            raise ValueError("Cần sử dụng fast tokenizer cho word_ids()")
    
    def initialize_model(self) -> None:
        """Khởi tạo mô hình"""
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name, 
            num_labels=len(self.config.labels), 
            id2label=self.id2label, 
            label2id=self.label2id
        )
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Tính toán metrics cho đánh giá"""
        predictions, labels = eval_pred
        preds = np.argmax(predictions, axis=-1)
        
        # Lọc bỏ các vị trí có nhãn -100
        pred_marks, gold_marks = [], []
        for pred_seq, label_seq in zip(preds, labels):
            mask = (label_seq != -100)
            pred_marks.extend(pred_seq[mask])
            gold_marks.extend(label_seq[mask])
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            gold_marks, pred_marks, average="binary", pos_label=self.label2id["BREAK"]
        )
        accuracy = np.mean(np.array(pred_marks) == np.array(gold_marks))
        
        return {
            "accuracy": accuracy,
            "precision": precision, 
            "recall": recall, 
            "f1": f1
        }
    
    def setup_trainer(self, train_dataset, eval_dataset) -> None:
        """Thiết lập trainer cho huấn luyện"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy="epoch",
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            save_strategy="epoch",
            save_total_limit=self.config.save_total_limit,
            max_grad_norm=self.config.max_grad_norm,
            logging_dir=self.config.logging_dir,
            logging_steps=50,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            no_cuda=not torch.cuda.is_available(),
        )
        
        # Khởi tạo callback
        self.metrics_callback = MetricsLoggerCallback()
        
        # Khởi tạo trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForTokenClassification(self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[self.metrics_callback]
        )
    
    def train(self, train_path: str, val_path: str) -> None:
        """Huấn luyện mô hình"""
        print("Bắt đầu huấn luyện...")
        
        # Tải tokenizer nếu chưa có
        if self.tokenizer is None:
            self.load_tokenizer("src/model/fast_tokenizer")
        
        # Khởi tạo mô hình nếu chưa có
        if self.model is None:
            self.initialize_model()
        
        # Khởi tạo processor
        self.processor = SentenceSplitterDatasetProcessor(self.tokenizer, self.config)
        
        # Tải và tiền xử lý dữ liệu
        processed_data = self.processor.load_and_preprocess(train_path, val_path)
        
        # Thiết lập trainer
        self.setup_trainer(processed_data["train"], processed_data["validation"])
        
        # Huấn luyện
        self.trainer.train()
        
        print("Huấn luyện hoàn tất!")
    
    def evaluate(self) -> Dict[str, float]:
        """Đánh giá mô hình trên tập validation"""
        if self.trainer is None:
            raise ValueError("Trainer chưa được khởi tạo. Vui lòng huấn luyện trước.")
        
        print("\nĐánh giá trên tập validation:")
        eval_result = self.trainer.evaluate()
        print(f"Kết quả đánh giá: {eval_result}")
        
        return eval_result
    
    def plot_training_curves(self, save_path: str = "training_curves.png") -> None:
        """Vẽ các biểu đồ huấn luyện"""
        if self.metrics_callback is None:
            print("Không có dữ liệu metrics để vẽ biểu đồ")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        ax1, ax2, ax3, ax4 = axes.flat
        
        # Biểu đồ 1: Loss curve
        if self.metrics_callback.train_loss:
            steps = range(len(self.metrics_callback.train_loss))
            ax1.plot(steps, self.metrics_callback.train_loss, 'b-', label="Train Loss", alpha=0.7)
        if self.metrics_callback.eval_loss:
            eval_steps = [i * len(self.metrics_callback.train_loss) // len(self.metrics_callback.eval_loss) 
                         for i in range(len(self.metrics_callback.eval_loss))]
            ax1.plot(eval_steps, self.metrics_callback.eval_loss, 'r-', label="Eval Loss", alpha=0.7)
        
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training / Validation Loss Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Biểu đồ 2: Validation F1
        if self.metrics_callback.eval_f1:
            epochs = range(1, len(self.metrics_callback.eval_f1) + 1)
            ax2.plot(epochs, self.metrics_callback.eval_f1, 'g-', marker='o', label="F1 Score")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("F1 Score")
            ax2.set_title("Validation F1 Score")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Biểu đồ 3: Precision-Recall-F1
        if (self.metrics_callback.eval_precision and 
            self.metrics_callback.eval_recall and 
            self.metrics_callback.eval_f1):
            epochs = range(1, len(self.metrics_callback.eval_f1) + 1)
            ax3.plot(epochs, self.metrics_callback.eval_precision, 'orange', 
                    marker='s', label="Precision", alpha=0.8)
            ax3.plot(epochs, self.metrics_callback.eval_recall, 'blue', 
                    marker='^', label="Recall", alpha=0.8)
            ax3.plot(epochs, self.metrics_callback.eval_f1, 'red', 
                    marker='o', label="F1", alpha=0.8)
            ax3.set_xlabel("Epoch")
            ax3.set_ylabel("Score")
            ax3.set_title("Precision - Recall - F1")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Biểu đồ 4: Accuracy
        if self.metrics_callback.eval_accuracy:
            epochs = range(1, len(self.metrics_callback.eval_accuracy) + 1)
            ax4.plot(epochs, self.metrics_callback.eval_accuracy, 'purple', 
                    marker='d', label="Accuracy")
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Accuracy")
            ax4.set_title("Validation Accuracy")
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Đã lưu biểu đồ tại: {save_path}")
    
    def analyze_thresholds(self, val_path: str, save_path: str = "f1_threshold_tuning.png") -> float:
        """Phân tích F1 theo các ngưỡng khác nhau"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Mô hình hoặc tokenizer chưa được khởi tạo")
        
        print("Phân tích F1 theo ngưỡng...")
        
        # Tải dữ liệu validation
        from datasets import load_dataset
        raw_val_data = load_dataset("json", data_files={"validation": val_path})["validation"]
        
        self.model.eval()
        thresholds = np.arange(0.1, 0.95, 0.05)
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for threshold in tqdm(thresholds):
            all_preds, all_labels = [], []
            
            for example in raw_val_data:
                words = example['text'].split()
                breaks_set = set(example['breaks'])
                
                enc = self.tokenizer(words, is_split_into_words=True, 
                                   return_tensors="pt", truncation=True, 
                                   max_length=self.config.max_length)
                
                with torch.no_grad():
                    logits = self.model(**enc).logits[0]
                    probs = torch.softmax(logits, dim=-1)
                
                word_ids = enc.word_ids(batch_index=0)
                current_word_preds = []
                current_word_labels = []
                
                for i, w_id in enumerate(word_ids):
                    if w_id is None: 
                        continue
                        
                    is_last_sub = (i + 1 == len(word_ids)) or (word_ids[i + 1] != w_id)
                    if is_last_sub:
                        # Nhãn thực tế
                        true_label = 1 if w_id in breaks_set else 0
                        current_word_labels.append(true_label)
                        
                        # Dự đoán
                        pred = 1 if probs[i, self.label2id['BREAK']].item() >= threshold else 0
                        current_word_preds.append(pred)
                
                all_labels.extend(current_word_labels)
                all_preds.extend(current_word_preds)
            
            # Tính metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="binary", zero_division=0
            )
            
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        # Tìm ngưỡng tốt nhất
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        self.best_threshold = best_threshold
        
        # Vẽ biểu đồ
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Ngưỡng tốt nhất: {best_threshold:.2f} với F1: {best_f1:.3f}")
        return best_threshold
    
    def save_model(self, model_dir: str = None, save_threshold: bool = True) -> None:
        """Lưu mô hình và tokenizer"""
        if model_dir is None:
            model_dir = self.config.output_dir
        
        print(f"\nLưu mô hình tại: {model_dir}")
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(model_dir, exist_ok=True)
        
        # Lưu mô hình và tokenizer
        if self.trainer is not None:
            self.trainer.save_model(model_dir)
        elif self.model is not None:
            self.model.save_pretrained(model_dir)
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(model_dir)
        
        # Lưu cấu hình
        config_dict = {
            "model_name": self.config.model_name,
            "labels": self.config.labels,
            "max_length": self.config.max_length,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "best_threshold": self.best_threshold
        }
        
        with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        # Lưu ngưỡng tốt nhất
        if save_threshold:
            with open(os.path.join(model_dir, "best_threshold.txt"), "w") as f:
                f.write(str(self.best_threshold))
        
        print("Đã lưu mô hình thành công!")
    
    def load_model(self, model_dir: str) -> None:
        """Tải mô hình đã huấn luyện"""
        print(f"Tải mô hình từ: {model_dir}")
        
        # Tải tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        
        # Tải cấu hình
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # Cập nhật cấu hình
            self.config.model_name = config_dict.get("model_name", self.config.model_name)
            self.config.labels = config_dict.get("labels", self.config.labels)
            self.config.max_length = config_dict.get("max_length", self.config.max_length)
            self.best_threshold = config_dict.get("best_threshold", 0.5)
            
            # Cập nhật ánh xạ nhãn
            self.label2id = config_dict.get("label2id", self.label2id)
            self.id2label = {int(k): v for k, v in config_dict.get("id2label", {}).items()}
        else:
            # Nếu không có config file, khởi tạo lại
            self._init_label_mappings()
        
        # Tải mô hình
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            num_labels=len(self.config.labels),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Tải ngưỡng tốt nhất
        threshold_path = os.path.join(model_dir, "best_threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                self.best_threshold = float(f.read().strip())
        
        print("Đã tải mô hình thành công!")
    
    def split_sentences(self, raw_text: str, threshold: float = None) -> List[str]:
        """Tách câu với ngưỡng tối ưu"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Mô hình hoặc tokenizer chưa được khởi tạo")
        
        if threshold is None:
            threshold = self.best_threshold
        
        words = raw_text.split()
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", 
                           truncation=True, max_length=self.config.max_length)
        
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1)
        
        pieces, cur = [], []
        word_ids = enc.word_ids(batch_index=0)
        
        for i, w_id in enumerate(word_ids):
            if w_id is None: 
                continue
                
            is_first_sub = (i == 0) or (word_ids[i-1] != w_id)
            if is_first_sub:
                cur.append(words[w_id])
                
            is_last_sub = (i + 1 == len(word_ids)) or (word_ids[i+1] != w_id)
            if is_last_sub:
                if probs[i, self.label2id["BREAK"]].item() >= threshold:
                    pieces.append(" ".join(cur).strip())
                    cur = []
        
        if cur:
            pieces.append(" ".join(cur).strip())
        
        return pieces
    
    def demo(self, text: str = None) -> None:
        """Chạy demo tách câu"""
        if text is None:
            text = "lan có 3 quả táo, mẹ cho thêm 4 quả táo nữa. Lan có tất cả bao nhiêu quả táo"
        
        print("\nDemo tách câu:")
        print(f"Văn bản gốc: {text}")
        
        result = self.split_sentences(text)
        for i, sent in enumerate(result, 1):
            print(f"Câu {i}: {sent}")


class SentenceSplitterTrainer:
    """Lớp quản lý quá trình huấn luyện và đánh giá"""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.splitter = SentenceSplitterModel(self.config)
        self.style_setup()
    
    def style_setup(self):
        """Thiết lập style cho biểu đồ"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        transformers.logging.set_verbosity_error()
    
    def run_training_pipeline(self, train_path: str, val_path: str) -> None:
        """Chạy toàn bộ pipeline huấn luyện"""
        # Huấn luyện mô hình
        self.splitter.train(train_path, val_path)
        
        # Đánh giá mô hình
        self.splitter.evaluate()
        
        # Vẽ biểu đồ huấn luyện
        print("\nVẽ biểu đồ quá trình huấn luyện...")
        self.splitter.plot_training_curves()
        
        # Phân tích ngưỡng
        print("\nPhân tích ngưỡng tối ưu...")
        best_threshold = self.splitter.analyze_thresholds(val_path)
        
        # Lưu mô hình
        self.splitter.save_model()
        
        # Demo
        self.splitter.demo()
        
        print("\nHoàn thành pipeline huấn luyện!")


def main():
    """Hàm chính"""
    # Cấu hình mô hình
    config = ModelConfig(
        model_name="vinai/phobert-base-v2",
        train_batch_size=16,
        eval_batch_size=16,
        num_epochs=3,
        output_dir="src/model/sent_split_model"
    )
    
    # Đường dẫn dữ liệu
    train_path = "data/train_separate.jsonl"
    val_path = "data/dev_separate.jsonl"
    
    # Khởi tạo và chạy trainer
    trainer = SentenceSplitterTrainer(config)
    trainer.run_training_pipeline(train_path, val_path)


if __name__ == "__main__":
    main()