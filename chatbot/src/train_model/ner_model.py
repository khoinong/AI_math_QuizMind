from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from tqdm.auto import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import classification_report

from transformers import (
AutoTokenizer,
AutoModelForTokenClassification,
TrainingArguments,
Trainer,
DataCollatorForTokenClassification,
TrainerCallback
)
from datasets import Dataset

@dataclass
class NERConfig:
    model_name: str = "vinai/phobert-base-v2"
    output_dir: str = "src/model/ner_model"
    logging_dir: str = "output/logs"
    train_batch_size: int = 16
    eval_batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 16
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_total_limit: int = 2
    logging_steps: int = 10
    use_cuda: bool = True
    # Nhãn NER
    label_list: List[str] = field(default_factory=lambda: [
        "O",
        "B-NUM", "I-NUM",
        "B-AGENT", "I-AGENT",
        "B-REL", "I-REL",
        "B-VALUE", "I-VALUE",
        "B-UNIT", "I-UNIT",
        "B-ATTRIBUTE", "I-ATTRIBUTE",
        "B-QUESTION", "I-QUESTION"
    ])
class NERDataProcessor:
        
    def __init__(self, config: NERConfig):
        self.config = config
        self.label2id = {label: i for i, label in enumerate(config.label_list)}
        self.id2label = {i: label for i, label in enumerate(config.label_list)}
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        """Thiết lập tokenizer"""
        self.tokenizer = tokenizer

    def read_conll_file(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """Đọc dữ liệu từ file CoNLL format"""
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

    def create_dataset(self, tokens: List[List[str]], labels: List[List[str]]) -> Dataset:
        """Tạo dataset từ tokens và labels"""
        return Dataset.from_dict({
            'tokens': tokens,
            'ner_tags': [[self.label2id[label] for label in seq] for seq in labels]
        })

    def tokenize_and_align_labels(self, examples: Dict[str, Any]) -> Dict[str, List[List[int]]]:
        """Tokenize và căn chỉnh nhãn với tokenization"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer chưa được thiết lập")
        
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for tokens, labels in zip(examples["tokens"], examples["ner_tags"]):
            input_ids = [self.tokenizer.cls_token_id]
            label_ids = [-100]
            attention_mask = [1]

            for word, label in zip(tokens, labels):
                word_tokens = self.tokenizer.tokenize(word)
                word_token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)

                input_ids.extend(word_token_ids)
                attention_mask.extend([1] * len(word_token_ids))

                label_ids.append(label)
                if len(word_token_ids) > 1:
                    label_ids.extend([-100] * (len(word_token_ids) - 1))

            input_ids.append(self.tokenizer.sep_token_id)
            attention_mask.append(1)
            label_ids.append(-100)

            # Cắt bớt nếu quá dài
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
class TrainingMetricsCallback(TrainerCallback):
    """Callback để theo dõi metrics trong quá trình huấn luyện"""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.train_epochs = []
        self.eval_epochs = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs and "epoch" in logs:
                self.train_losses.append(logs["loss"])
                self.train_epochs.append(logs["epoch"])
            if "eval_loss" in logs and "epoch" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_epochs.append(logs["epoch"])
class NERTrainer:
    """Lớp chính để huấn luyện mô hình NER"""

    def __init__(self, config: NERConfig = None):
        self.config = config or NERConfig()
        self.data_processor = NERDataProcessor(self.config)
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.metrics_callback = TrainingMetricsCallback()
        
        # Thiết lập CUDA
        self.config.use_cuda = not self.config.no_cuda if hasattr(self.config, 'no_cuda') else True

    def load_tokenizer(self):
        """Tải tokenizer"""
        print(f"Đang tải tokenizer từ {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=False)
        self.data_processor.set_tokenizer(self.tokenizer)

    def load_model(self):
        """Tải mô hình"""
        print(f"Đang tải mô hình từ {self.config.model_name}...")
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.config.label_list),
            id2label=self.data_processor.id2label,
            label2id=self.data_processor.label2id
        )

    def prepare_datasets(self, train_path: str, val_path: str) -> Tuple[Dataset, Dataset]:
        """Chuẩn bị dataset từ các file CoNLL"""
        print("Đang đọc dữ liệu...")
        
        # Đọc dữ liệu
        train_tokens, train_labels = self.data_processor.read_conll_file(train_path)
        val_tokens, val_labels = self.data_processor.read_conll_file(val_path)
        
        print(f"Số lượng mẫu huấn luyện: {len(train_tokens)}")
        print(f"Số lượng mẫu validation: {len(val_tokens)}")
        
        # Tạo dataset
        train_dataset = self.data_processor.create_dataset(train_tokens, train_labels)
        val_dataset = self.data_processor.create_dataset(val_tokens, val_labels)
        
        # Tokenize
        print("Đang tokenize dữ liệu...")
        tokenized_train = train_dataset.map(
            self.data_processor.tokenize_and_align_labels, 
            batched=True
        )
        tokenized_val = val_dataset.map(
            self.data_processor.tokenize_and_align_labels, 
            batched=True
        )
        
        return tokenized_train, tokenized_val

    def setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset):
        """Thiết lập trainer cho huấn luyện"""
        # Training arguments
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
            logging_steps=self.config.logging_steps,
            load_best_model_at_end=True,
            no_cuda=not self.config.use_cuda,
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[self.metrics_callback]
        )

    def train(self, resume_from_checkpoint: bool = False):
        """Huấn luyện mô hình"""
        print("Bắt đầu huấn luyện...")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("Huấn luyện hoàn tất!")

    def evaluate(self) -> Dict[str, Any]:
        """Đánh giá mô hình trên tập validation"""
        print("\nĐánh giá mô hình...")
        
        # Dự đoán
        predictions = self.trainer.predict(self.trainer.eval_dataset)
        preds = np.argmax(predictions.predictions, axis=2)
        
        # Chuyển đổi về nhãn
        true_labels = []
        pred_labels = []
        
        for i, label_seq in enumerate(predictions.label_ids):
            true_line = []
            pred_line = []
            for j, label_id in enumerate(label_seq):
                if label_id != -100:
                    true_line.append(self.data_processor.id2label[label_id])
                    pred_line.append(self.data_processor.id2label[preds[i][j]])
            true_labels.append(true_line)
            pred_labels.append(pred_line)
        
        return {
            "predictions": predictions,
            "pred_labels": pred_labels,
            "true_labels": true_labels,
            "preds": preds
        }

    def save_plots(self, eval_results: Dict[str, Any]):
        """Lưu các biểu đồ đánh giá"""
        plot_dir = os.path.join(self.config.output_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # 1. Loss curve
        self._plot_loss_curve(plot_dir)
        
        # 2. Macro P/R/F1
        self._plot_macro_metrics(plot_dir, eval_results)
        
        # 3. Per-label F1
        self._plot_per_label_f1(plot_dir, eval_results)

    def _plot_loss_curve(self, plot_dir: str):
        """Vẽ biểu đồ loss"""
        if not self.metrics_callback.train_losses:
            print("Không có dữ liệu loss để vẽ biểu đồ")
            return
        
        plt.figure(figsize=(8, 5))
        
        # Vẽ train loss
        if self.metrics_callback.train_losses:
            plt.plot(self.metrics_callback.train_epochs, 
                    self.metrics_callback.train_losses, 
                    label="train_loss")
        
        # Vẽ eval loss
        if self.metrics_callback.eval_losses:
            plt.plot(self.metrics_callback.eval_epochs, 
                    self.metrics_callback.eval_losses, 
                    label="eval_loss")
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training / Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        loss_curve_path = os.path.join(plot_dir, "loss_curve.png")
        plt.savefig(loss_curve_path)
        plt.close()
        print(f"[VIS] Đã lưu biểu đồ loss tại: {loss_curve_path}")

    def _plot_macro_metrics(self, plot_dir: str, eval_results: Dict[str, Any]):
        """Vẽ biểu đồ macro metrics"""
        flat_true = []
        flat_pred = []
        
        for tseq, pseq in zip(eval_results["true_labels"], eval_results["pred_labels"]):
            flat_true.extend(tseq)
            flat_pred.extend(pseq)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            flat_true, flat_pred, average="macro"
        )
        
        plt.figure(figsize=(7, 5))
        plt.bar(["Precision", "Recall", "F1"], [precision, recall, f1])
        plt.title("Macro Precision - Recall - F1")
        plt.ylim(0, 1)
        plt.tight_layout()
        
        macro_metrics_path = os.path.join(plot_dir, "precision_recall_f1.png")
        plt.savefig(macro_metrics_path)
        plt.close()
        print(f"[VIS] Đã lưu biểu đồ macro metrics tại: {macro_metrics_path}")

    def _plot_per_label_f1(self, plot_dir: str, eval_results: Dict[str, Any]):
        """Vẽ biểu đồ F1 cho từng nhãn"""
        flat_true = []
        flat_pred = []
        
        for tseq, pseq in zip(eval_results["true_labels"], eval_results["pred_labels"]):
            flat_true.extend(tseq)
            flat_pred.extend(pseq)
        
        per_label_precision, per_label_recall, per_label_f1, _ = precision_recall_fscore_support(
            flat_true, flat_pred, labels=self.config.label_list, average=None
        )
        
        plt.figure(figsize=(12, 5))
        plt.bar(self.config.label_list, per_label_f1)
        plt.title("Per-label F1 Score")
        plt.ylabel("F1 Score")
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        per_label_path = os.path.join(plot_dir, "per_label_f1.png")
        plt.savefig(per_label_path)
        plt.close()
        print(f"[VIS] Đã lưu biểu đồ per-label F1 tại: {per_label_path}")

    def print_classification_report(self, eval_results: Dict[str, Any]):
        """In báo cáo phân loại"""
        print("\n" + "="*50)
        print("BÁO CÁO PHÂN LOẠI")
        print("="*50)
        
        report = classification_report(
            eval_results["true_labels"], 
            eval_results["pred_labels"]
        )
        print(report)

    def save_model(self):
        """Lưu mô hình đã huấn luyện"""
        if self.trainer is not None:
            print(f"\nLưu mô hình tại: {self.config.output_dir}")
            self.trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            print("Đã lưu mô hình thành công!")

    def run_training_pipeline(self, train_path: str, val_path: str):
        """Chạy toàn bộ pipeline huấn luyện"""
        # 1. Tải tokenizer và mô hình
        self.load_tokenizer()
        self.load_model()
        
        # 2. Chuẩn bị dữ liệu
        train_dataset, val_dataset = self.prepare_datasets(train_path, val_path)
        
        # 3. Thiết lập trainer
        self.setup_trainer(train_dataset, val_dataset)
        
        # 4. Huấn luyện
        self.train(resume_from_checkpoint=False)
        
        # 5. Đánh giá
        eval_results = self.evaluate()
        
        # 6. Lưu biểu đồ
        self.save_plots(eval_results)
        
        # 7. In báo cáo
        self.print_classification_report(eval_results)
        
        # 8. Lưu mô hình
        self.save_model()
        
        print("\n" + "="*50)
        print("HOÀN THÀNH HUẤN LUYỆN MÔ HÌNH NER")
        print("="*50)
def main():
# Cấu hình
    config = NERConfig(
    model_name="vinai/phobert-base-v2",
    output_dir="src/model/ner_model",
    train_batch_size=16,
    eval_batch_size=16,
    num_epochs=16,
    logging_steps=10,
    use_cuda=False # Đặt True nếu có GPU
    )

    # Đường dẫn dữ liệu
    train_path = "data/train.conll"
    val_path = "data/dev.conll"

    # Khởi tạo và chạy pipeline
    ner_trainer = NERTrainer(config)
    ner_trainer.run_training_pipeline(train_path, val_path)
if __name__ == "__main__":
    main()