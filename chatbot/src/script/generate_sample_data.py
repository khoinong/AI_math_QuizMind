# Revised MathProblemGenerator with expanded subjects and units
# (Replace majority of subjects and measurement units as user requested)

import json
import random
from typing import List, Dict

class MathProblemGenerator:
    def __init__(self):
        self.examples = []

    def add_examples_from_sentences(self, sentences: List[str]) -> None:
        if random.random() < 0.35:
            random.shuffle(sentences)

        synonyms = {
            "hỏi": ["tính", "cho biết", "tính xem", "vậy thì", "vậy hỏi"],
            "còn lại": ["còn", "còn bao nhiêu", "còn lại tất cả"],
            "tất cả": ["tổng cộng", "gộp lại", "lại là"],
        }

        new_sentences = []
        for s in sentences:
            for key, syns in synonyms.items():
                if key in s and random.random() < 0.4:
                    s = s.replace(key, random.choice(syns))
            new_sentences.append(s)

        sentences = new_sentences

        breaks = []
        current_pos = -1
        for i in range(len(sentences)-1):
            current_pos += len(sentences[i].split())
            breaks.append(current_pos)

        cleaned = [sent.strip().rstrip('.,!?') for sent in sentences]
        variants = [
            ",".join(cleaned),
            ". ".join(cleaned),
            " ".join(cleaned),
        ]

        for text in variants:
            self.examples.append({"text": text, "breaks": breaks})

    # Expanded subjects and items
    SUBJECTS = [
        "anh", "chị", "bé An", "bé Na", "ông", "bà", "cô giáo", "thầy giáo",
        "bác sĩ", "nông dân", "công nhân", "tài xế", "thợ may", "học sinh",
        "sinh viên", "bảo vệ", "lính cứu hỏa", "phi công", "nhân viên thư viện"
    ]

    OBJECTS = [
        "quả xoài", "quả táo", "quả ổi", "quả dưa", "quả mít", "bánh mì",
        "chai nước", "cốc trà", "bút bi", "bút chì", "vở ghi", "tập giấy",
        "chiếc dép", "đôi tất", "lon nước ngọt", "hộp phấn", "cục tẩy"
    ]

    UNITS = [
        "kg", "g", "lít", "ml", "cm", "mm", "m", "tờ", "cuốn", "bịch",
        "hộp", "thùng", "bao", "viên", "miếng", "gói"
    ]

    def generate_ownership_templates(self) -> List[List[str]]:
        base = []
        for subj in self.SUBJECTS:
            for obj in self.OBJECTS:
                base.append([
                    f"{subj} có {{n1}} {obj}",
                    f"được cho thêm {{n2}} {obj.split()[0]}",
                    f"hỏi {subj} có tất cả bao nhiêu {obj.split()[0]}"
                ])

                base.append([
                    f"{subj} có {{n1}} {obj}",
                    f"cho đi {{n2}} {obj.split()[0]}",
                    f"{subj} còn lại bao nhiêu {obj.split()[0]}"
                ])
        return base

    def generate_ratio_templates(self) -> List[List[str]]:
        base = []
        for unit in self.UNITS:
            base.append([
                f"một hộp có {{n1}} {unit}",
                f"{{n2}} hộp như vậy có mấy {unit}"
            ])
        return base

    def generate_comparison_templates(self) -> List[List[str]]:
        base = []
        for a in self.SUBJECTS:
            for b in self.SUBJECTS:
                if a != b:
                    base.append([
                        f"{a} có {{n1}} {random.choice(self.OBJECTS)}",
                        f"{b} nhiều hơn {a} {{n2}} cái",
                        f"hỏi {b} có mấy cái"
                    ])
        return base

    def generate_division_templates(self) -> List[List[str]]:
        base = []
        for obj in self.OBJECTS:
            for people in [2,3,4,5]:
                base.append([
                    f"có {{n1}} {obj}",
                    f"chia đều cho {people} người",
                    f"mỗi người được bao nhiêu {obj.split()[0]}"
                ])
        return base

    def generate_mixed_templates(self) -> List[List[str]]:
        places = ["trên bàn", "trong tủ", "trong hộp", "dưới sàn", "trong giỏ"]
        base = []
        for place in places:
            for obj in self.OBJECTS:
                base.append([
                    f"{place} có {{n1}} {obj}",
                    f"lấy đi {{n2}} {obj.split()[0]}",
                    f"còn lại bao nhiêu {obj.split()[0]}"
                ])
        return base

    def validate_breaks(self, text: str, breaks: List[int]) -> bool:
        words = text.split()
        if not breaks:
            return True
        return all(b < len(words) for b in breaks)

    def generate_problems(self) -> List[Dict]:
        templates = [
            self.generate_ownership_templates(),
            self.generate_ratio_templates(),
            self.generate_comparison_templates(),
            self.generate_division_templates(),
            self.generate_mixed_templates()
        ]

        for template_list in templates:
            for _ in range(400):
                template = random.choice(template_list)
                n1 = random.randint(1, 150)
                n2 = random.randint(1, max(1, n1//2))
                sents = [s.format(n1=n1, n2=n2) for s in template]
                self.add_examples_from_sentences(sents)

        valid = []
        for ex in self.examples:
            if self.validate_breaks(ex["text"], ex["breaks"]):
                valid.append(ex)
        self.examples = valid
        return valid

def create_enhanced_variants(example: Dict) -> List[Dict]:
    variants = []
    text = example["text"]
    breaks = example["breaks"]
    words = text.split()

    # original
    variants.append(example)

    # random no-punct
    if random.random() < 0.7:
        variants.append({"text": " ".join(words), "breaks": breaks})

    # random comma insert
# random comma insert
    for b in breaks[:3]:
        if b < len(words) and random.random() < 0.2:
            new = words.copy()
            new[b] = new[b] + ","
            variants.append({"text": " ".join(new), "breaks": breaks})

    # weird punctuation mix -> make it minimal
    if random.random() < 0.2:
        weird = " ".join(words) + "."
        variants.append({"text": weird, "breaks": breaks})


    return variants


def save_jsonl(data: List[Dict], filename: str) -> None:
    """Save data to JSONL file with error handling"""
    import os
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Successfully saved {len(data)} examples to {filename}")
    except Exception as e:
        print(f"Error saving to {filename}: {e}")

def main():
    """Main execution function"""
    # Initialize generator
    generator = MathProblemGenerator()
    
    # Generate data
    print("Generating math problems...")
    train_data = generator.generate_problems()
    
    # Create variants
    print("Creating variants...")
    enhanced_train_data = []
    for example in train_data:
        variants = create_enhanced_variants(example)
        enhanced_train_data.extend(variants)
    
    # Split train/dev
    random.shuffle(enhanced_train_data)
    split_idx = int(0.8 * len(enhanced_train_data))
    final_train_data = enhanced_train_data[:split_idx]
    final_dev_data = enhanced_train_data[split_idx:split_idx + 600]  # Limit dev size
    
    # Save data
    save_jsonl(final_train_data, "data/train_enhanced.jsonl")
    save_jsonl(final_dev_data, "data/dev_enhanced.jsonl")
    
    # Print statistics
    print(f"\n=== Generation Statistics ===")
    print(f"Total training examples: {len(final_train_data)}")
    print(f"Total development examples: {len(final_dev_data)}")
    
    if final_train_data:
        print(f"\nExample from training set:")
        print(json.dumps(final_train_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()