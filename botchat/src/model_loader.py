# model_loader.py
import torch
import json
import tensorflow as tf
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    RobertaTokenizerFast,
    AutoTokenizer
)
from seqeval.metrics import classification_report
import sys
import codecs
import argparse
import numpy as np
import os
import re
import string


class ClassificationModel:
    """Mô hình phân loại bài toán sử dụng PhoBERT"""
    
    LABEL_NAMES = ["basic", "basic_word", "ownership", "ratio", "comparison"]

    def __init__(self, model_path):
        """Khởi tạo model phân loại bài toán

        Args:
            model_path: Đường dẫn đến thư mục chứa model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

        # Prefer loading a TensorFlow SavedModel if present (training script saves Keras SavedModel),
        # otherwise fallback to a HuggingFace transformers model (PyTorch)
        self.is_tf_model = False
        saved_model_dir = None
        # Common saved_model layout: <model_path>/saved_model/... (saved via model.save(saved_path))
        if os.path.isdir(os.path.join(model_path, "saved_model")):
            saved_model_dir = os.path.join(model_path, "saved_model")
        # Also accept if user passed directly the saved_model directory
        elif os.path.isdir(model_path) and (
            os.path.exists(os.path.join(model_path, "saved_model.pb")) or
            os.path.isdir(os.path.join(model_path, "variables"))
        ):
            saved_model_dir = model_path

        if saved_model_dir:
            try:
                # Load Keras SavedModel (TF) for inference
                # This returns a keras.Model with a callable signature matching the training inputs
                self.tf_model = tf.keras.models.load_model(saved_model_dir)
                self.is_tf_model = True
                self.tf_use_signature = False
            except Exception as e:
                # Try loading with low-level saved_model loader and use serving_default signature
                try:
                    loaded = tf.saved_model.load(saved_model_dir)
                    if hasattr(loaded, "signatures") and "serving_default" in loaded.signatures:
                        self.saved_signature = loaded.signatures["serving_default"]
                        self.is_tf_model = True
                        self.tf_use_signature = True
                    else:
                        raise RuntimeError("SavedModel has no serving_default signature")
                except Exception as e2:
                    raise RuntimeError(f"Failed to load TF SavedModel from {saved_model_dir}: {e}; fallback failed: {e2}") from e2
        else:
            # Fallback: try to load a transformers model (PyTorch)
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                num_labels=len(self.LABEL_NAMES)
            ).to(self.device)
        
    def predict(self, text):
        """Dự đoán loại bài toán từ văn bản
        
        Args:
            text: Văn bản cần phân loại

        Returns:
            tuple: (tên_nhãn, dict xác suất từng nhãn)
        """
        # Tokenize và encode. Use TF tensors if using TF model, otherwise PyTorch tensors.
        if getattr(self, "is_tf_model", False):
            enc = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="tf",
            )

            # Build input dict matching training signature
            inputs = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

            # If we loaded a Keras model, call it directly. Otherwise call the SavedModel signature.
            if getattr(self, "tf_use_signature", False):
                # signature expects TF Tensors; convert if needed
                sig_inputs = {k: tf.convert_to_tensor(v) for k, v in inputs.items()}
                sig_out = self.saved_signature(**sig_inputs)
                # signature returns a dict of outputs; take first tensor
                first_out = list(sig_out.values())[0]
                probs = first_out.numpy()[0]
            else:
                preds = self.tf_model(inputs, training=False)
                probs = preds.numpy()[0]
        else:
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=96,
                return_tensors="pt"
            ).to(self.device)

            # Dự đoán PyTorch HF model
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

        pred_idx = int(np.argmax(probs))

        # Chuyển về định dạng dễ đọc
        prob_dict = {name: float(f"{p:.3f}") for name, p in zip(self.LABEL_NAMES, probs)}

        return self.LABEL_NAMES[pred_idx], prob_dict

    def classify_problem(self, text):
        """
        Phương thức gọi model phân loại bài toán
        Args:
            text: Văn bản cần phân loại
        Returns:
            dict: Kết quả phân loại với loại bài toán và xác suất
        """
        problem_type, probabilities = self.predict(text)
        return {
            "problem_type": problem_type,
            "probabilities": probabilities,
            "text": text
        }


class SeparateModel:
    def __init__(self, model_path, threshold=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.label2id = {"CONT": 0, "BREAK": 1}
        self.id2label = {0: "CONT", 1: "BREAK"}

        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

    def separate_text(self, text):
        """Tách văn bản thành các câu dựa hoàn toàn vào model (không tiền xử lý)."""
        words = text.split()
        if not words:
            return [text]

        # Encode & chạy model
        enc = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

        sentences, cur = [], []
        enc_cpu = self.tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=512)
        word_ids_seq = enc_cpu.word_ids(batch_index=0)

        for i, w_id in enumerate(word_ids_seq):
            if w_id is None:
                continue
            is_first_sub = (i == 0) or (word_ids_seq[i-1] != w_id)
            if is_first_sub:
                cur.append(words[w_id])

            is_last_sub = (i + 1 == len(word_ids_seq)) or (word_ids_seq[i + 1] != w_id)
            if is_last_sub:
                prob_break = probs[i, self.label2id["BREAK"]].item()
                if prob_break >= self.threshold:
                    sentences.append(" ".join(cur).strip())
                    cur = []

        if cur:
            sentences.append(" ".join(cur).strip())

        # 🧹 Hậu xử lý: loại bỏ dấu câu ", . ? !" ở cuối mỗi câu
        cleaned_sentences = [
            re.sub(r'[\s,\.?!]+$', '', s.strip()) for s in sentences if s.strip()
        ]

        return cleaned_sentences

    def split_sentences(self, text):
        """Phương thức gọi model tách câu"""
        sentences = self.separate_text(text)
        return {
            "original_text": text,
            "sentences": sentences,
            "sentence_count": len(sentences)
        }



class NERModel:
    def __init__(self, model_path, label_list):
        self.label_list = label_list
        self.id2label = {i: label for i, label in enumerate(label_list)}
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model và tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

    @staticmethod
    def strip_punctuation(s):
        """Loại bỏ dấu câu nhưng giữ lại các toán tử + - * / ="""
        # Giữ lại các toán tử và dấu bằng
        keep_chars = "+-*/="
        # Loại bỏ các dấu câu khác
        no_punct = re.sub(r'[^\w\s' + re.escape(keep_chars) + ']', '', s)
        no_punct = re.sub(r'\s+', ' ', no_punct).strip()
        return no_punct

    def postprocess(self, tokens, labels):
        """
        Hậu xử lý kết quả NER:
        - Ghép các token liên tiếp cùng nhãn (nếu cần)
        - Loại bỏ nhãn không hợp lệ (nếu có)
        - Chuẩn hóa nhãn (nếu cần)
        """
        processed_tokens = []
        processed_labels = []
        prev_label = None
        buffer = []
        for token, label in zip(tokens, labels):
            # Ví dụ: ghép các token liên tiếp cùng nhãn (B-*, I-*)
            if label.startswith("I-") and prev_label and prev_label[2:] == label[2:]:
                buffer.append(token)
            else:
                if buffer:
                    processed_tokens.append(" ".join(buffer))
                    processed_labels.append(prev_label)
                    buffer = []
                buffer = [token]
                prev_label = label
        if buffer:
            processed_tokens.append(" ".join(buffer))
            processed_labels.append(prev_label)
        # Loại bỏ nhãn không hợp lệ (ví dụ: None)
        final_tokens = [t for t, l in zip(processed_tokens, processed_labels) if l is not None]
        final_labels = [l for l in processed_labels if l is not None]
        return final_tokens, final_labels

    def predict(self, text):
        """Dự đoán nhãn NER cho một câu (giữ nguyên toàn bộ ký tự đầu vào)."""
        words = text.split()

        tokens, word_ids = [], []
        for word_idx, word in enumerate(words):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            word_ids.extend([word_idx] * len(word_tokens))

        input_ids = [self.tokenizer.cls_token_id] + \
                    self.tokenizer.convert_tokens_to_ids(tokens) + \
                    [self.tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        inputs = {
            "input_ids": torch.tensor([input_ids]).to(self.device),
            "attention_mask": torch.tensor([attention_mask]).to(self.device)
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2).cpu().numpy()[0][1:len(tokens) + 1]

        final_tokens, final_labels = [], []
        prev_word_idx = -1
        for token, word_idx, pred in zip(tokens, word_ids, predictions):
            if word_idx != prev_word_idx:
                final_tokens.append(token[2:] if token.startswith("##") else token)
                final_labels.append(self.id2label[pred])
                prev_word_idx = word_idx

        return final_tokens, final_labels

    def extract_entities(self, text):
        """
        Phương thức gọi model NER để trích xuất thực thể
        Args:
            text: Văn bản cần trích xuất
        Returns:
            dict: Kết quả trích xuất với tokens và labels
        """
        tokens, labels = self.predict(text)
        return {
            "text": text,
            "tokens": tokens,
            "labels": labels,
            "entities": list(zip(tokens, labels))
        }

    def evaluate(self, test_file_path, show_examples=3):
        """Đánh giá mô hình trên tập test định dạng CoNLL"""

        def read_conll_file(file_path):
            tokens, labels = [], []
            current_tokens, current_labels = [], []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 2:
                            current_tokens.append(parts[0])
                            current_labels.append(parts[-1])
                    else:
                        if current_tokens:
                            tokens.append(current_tokens)
                            labels.append(current_labels)
                            current_tokens, current_labels = [], []
                if current_tokens:
                    tokens.append(current_tokens)
                    labels.append(current_labels)
            return tokens, labels

        test_tokens, true_labels = read_conll_file(test_file_path)
        all_predicted, all_true = [], []

        for i, (sentence_tokens, sentence_true) in enumerate(zip(test_tokens, true_labels)):
            text = " ".join(sentence_tokens)
            try:
                _, predicted = self.predict(text)

                min_len = min(len(predicted), len(sentence_true))
                all_predicted.append(predicted[:min_len])
                all_true.append(sentence_true[:min_len])

                if i < show_examples:
                    print(f"\nCâu {i + 1}:")
                    print("Token\t\tTrue\t\tPredicted")
                    print("-" * 40)
                    for j in range(min_len):
                        print(f"{sentence_tokens[j]}\t\t{sentence_true[j]}\t\t{predicted[j]}")
            except Exception as e:
                print(f"Lỗi xử lý câu {i+1}: {e}")

        if all_predicted and all_true:
            print("\n" + "=" * 60)
            print("BÁO CÁO ĐÁNH GIÁ")
            print("=" * 60)
            print(classification_report(all_true, all_predicted))
        else:
            print("Không có dự đoán nào để đánh giá.")


def setup_args():
    parser = argparse.ArgumentParser(description="Test Models")
    parser.add_argument("--ner_model_path", type=str, default="src/model/ner_model")
    parser.add_argument("--separate_model_path", type=str, default="src/model/sent_split_model")
    parser.add_argument("--classification_model_path", type=str, default="src/model/classification_model")
    parser.add_argument("--test_file", type=str, help="Đường dẫn đến file test CoNLL")
    parser.add_argument("--text", type=str, help="Văn bản để test")
    parser.add_argument("--output_file", type=str, help="Ghi kết quả vào file")
    return parser.parse_args()


def setup_stdout(output_file=None):
    """Fix encoding stdout + redirect nếu có file output"""
    if sys.stdout.encoding != "UTF-8":
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")

    original_stdout = sys.stdout
    if output_file:
        sys.stdout = open(output_file, "w", encoding="utf-8")
    return original_stdout


def get_label_list():
    """Trả về danh sách nhãn NER"""
    return [
        "O",
        "B-NUM", "I-NUM",
        "B-AGENT", "I-AGENT",
        "B-REL", "I-REL",
        "B-VALUE", "I-VALUE",
        "B-UNIT", "I-UNIT",
        "B-ATTRIBUTE", "I-ATTRIBUTE",
        "B-QUESTION", "I-QUESTION"
    ]


def run_test(args):
    # Khởi tạo các model
    label_list = get_label_list()
    ner_model = NERModel(args.ner_model_path, label_list)
    separate_model = SeparateModel(args.separate_model_path)
    classification_model = ClassificationModel(args.classification_model_path)
    
    print(f"Đã tải NER model từ {args.ner_model_path}")
    print(f"Đã tải Separate model từ {args.separate_model_path}")
    print(f"Đã tải Classification model từ {args.classification_model_path}")
    print(f"Đang sử dụng device: {ner_model.device}")

    def analyze_text(text, is_sample=False):
        # Sử dụng phương thức mới để phân loại bài toán
        classification_result = classification_model.classify_problem(text)
        print(f"\nKẾT QUẢ PHÂN LOẠI{' (Sample)' if is_sample else ''}:")
        print("-" * 60)
        print(f"Loại bài toán: {classification_result['problem_type']}")
        print("Xác suất từng loại:")
        max_type_len = max(len(t) for t in classification_result['probabilities'].keys())
        for t, p in classification_result['probabilities'].items():
            print(f"  {t:<{max_type_len}} : {p:.3f}")

        # Sử dụng phương thức mới để tách câu
        separation_result = separate_model.split_sentences(text)
        
        # Hàm strip_punctuation mới để giữ lại toán tử khi hiển thị
        def strip_punctuation_keep_math(s):
            keep_chars = "+-*/="
            no_punct = re.sub(r'[^\w\s' + re.escape(keep_chars) + ']', '', s)
            no_punct = re.sub(r'\s+', ' ', no_punct).strip()
            return no_punct

        sentences = separation_result['sentences']


        print(f"\nKẾT QUẢ TÁCH CÂU{' (Sample)' if is_sample else ''}:")
        print("-" * 60)
        print(f"Số lượng câu: {separation_result['sentence_count']}")
        for i, sentence in enumerate(sentences, 1):
            print(f"[{i}] {sentence}")

        print(f"\nKẾT QUẢ PHÂN TÍCH NER{' (Sample)' if is_sample else ''}:")
        print("-" * 60)
        for i, sentence in enumerate(sentences, 1):
            # sentence đã được làm sạch nhưng vẫn giữ toán tử
            clean_sentence = sentence
            print(f"\nCâu [{i}]: {clean_sentence}")
            print("-" * 40)
            
            # Sử dụng phương thức mới để trích xuất thực thể
            ner_result = ner_model.extract_entities(clean_sentence)
            
            # Tính độ rộng cột hiển thị
            max_token_len = max(len(token) for token in ner_result['tokens']) if ner_result['tokens'] else 0
            token_col = max(max_token_len + 2, 10)
            label_col = 15
            total_width = token_col + label_col + 5
            
            # In header
            print("┌" + "─" * token_col + "┬" + "─" * label_col + "┐")
            print(f"│{'Token':<{token_col}}│{'Nhãn':<{label_col}}│")
            print("├" + "─" * token_col + "┼" + "─" * label_col + "┤")
            
            # In nội dung
            for token, label in ner_result['entities']:
                # Bỏ @@ ở cuối token nếu có
                clean_token = token[:-2] if token.endswith("@@") else token
                print(f"│{clean_token:<{token_col}}│{label:<{label_col}}│")
            
            print("└" + "─" * token_col + "┴" + "─" * label_col + "┘")

    if args.test_file:
        ner_model.evaluate(args.test_file)
    elif args.text:
        analyze_text(args.text)
    else:
        sample_text = "1.3 + 4 * 5 - 6 / 2 = ?"
        analyze_text(sample_text, is_sample=True)


def main():
    args = setup_args()
    original_stdout = setup_stdout(args.output_file)

    run_test(args)

    if args.output_file:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Đã ghi kết quả vào {args.output_file}")


# Thêm các hàm tiện ích để sử dụng từ bên ngoài
def create_classification_model(model_path):
    """Tạo và trả về model phân loại"""
    return ClassificationModel(model_path)


def create_separate_model(model_path, threshold=0.5):
    """Tạo và trả về model tách câu"""
    return SeparateModel(model_path, threshold)


def create_ner_model(model_path, label_list=None):
    """Tạo và trả về model NER"""
    if label_list is None:
        label_list = get_label_list()
    return NERModel(model_path, label_list)


if __name__ == "__main__":
    main()