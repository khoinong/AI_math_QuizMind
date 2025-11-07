import json
import random
from typing import List, Dict

class MathProblemGenerator:
    def __init__(self):
        self.examples = []
        
    def add_examples_from_sentences(self, sentences: List[str]) -> None:
        """Enhanced version with better period handling"""
        # Compute breaks based on sentence word counts
        breaks = []
        current_pos = -1
        for i in range(len(sentences)-1):
            current_pos += len(sentences[i].split())
            breaks.append(current_pos)

        # Clean sentences by removing existing punctuation
        cleaned_sentences = []
        for sent in sentences:
            clean_sent = sent.strip().rstrip('.,!?')
            cleaned_sentences.append(clean_sent)

        # Variant 1: Comma-separated
        comma_text = " , ".join(cleaned_sentences)
        self.examples.append({"text": comma_text, "breaks": breaks})

        # Variant 2: Period-separated (proper sentence formatting)
        period_text = ". ".join(cleaned_sentences) + "."
        self.examples.append({"text": period_text, "breaks": breaks})

        # Variant 3: No punctuation
        no_punct = " ".join(" ".join(cleaned_sentences).split())
        self.examples.append({"text": no_punct, "breaks": breaks})

    def generate_ownership_templates(self) -> List[List[str]]:
        """Generate ownership problem templates"""
        base_templates = [
            ["lan có {n1} quả táo", "mẹ cho thêm {n2} quả nữa", "hỏi lan có mấy quả táo"],
            ["mai có {n1} cái kẹo", "cho bạn {n2} cái", "hỏi mai còn lại mấy cái kẹo"],
            ["trong lớp có {n1} học sinh", "nghỉ học {n2} bạn", "hỏi lớp còn lại bao nhiêu học sinh"],
            ["an có {n1} viên bi", "cho bạn {n2} viên", "hỏi an còn lại bao nhiêu viên"],
            ["tủ có {n1} quyển sách", "mua thêm {n2} quyển nữa", "hỏi tủ có tất cả bao nhiêu quyển sách"],
            ["trường có {n1} phòng học", "xây thêm {n2} phòng", "hỏi trường có tất cả bao nhiêu phòng"],
            ["thư viện có {n1} ghế", "mượn đi {n2} ghế", "hỏi thư viện còn lại bao nhiêu ghế"],
            ["siêu thị có {n1} xe đẩy", "cho mượn {n2} xe", "hỏi siêu thị còn lại bao nhiêu xe"],
            ["vườn có {n1} cây xoài", "chết {n2} cây", "hỏi vườn còn lại bao nhiêu cây"],
            ["bể có {n1} con cá", "bán đi {n2} con", "hỏi bể còn lại bao nhiêu con"]
        ]
        
        # Add more templates programmatically
        subjects = ["lan", "nam", "an", "mai", "huy", "hoa", "tu", "linh", "binh", "thu"]
        objects = ["quả cam", "quả táo", "cái kẹo", "chiếc bánh", "quyển vở", "cây bút", "viên bi"]
        
        for subj in subjects:
            for obj in objects:
                base_templates.append([
                    f"{subj} có {{n1}} {obj}",
                    f"cho {subj} {{n2}} {obj.split()[0]}",
                    f"hỏi {subj} còn lại bao nhiêu {obj.split()[0]}"
                ])
        
        return base_templates

    def generate_ratio_templates(self) -> List[List[str]]:
        """Generate ratio problem templates"""
        base_templates = [
            ["một con gà có {n1} chân", "{n2} con gà có mấy chân"],
            ["một xe đạp có {n1} bánh", "{n2} xe đạp có mấy bánh"],
            ["một bàn học có {n1} chân", "{n2} bàn học có mấy chân"],
            ["một con mèo có {n1} chân", "{n2} con mèo có mấy chân"],
            ["một bông hoa có {n1} cánh", "{n2} bông hoa có mấy cánh"],
            ["một lớp học có {n1} bàn", "{n2} lớp học có mấy bàn"],
            ["một xe bus có {n1} chỗ ngồi", "{n2} xe bus có mấy chỗ ngồi"],
            ["một cây bút có {n1} ngòi", "{n2} cây bút có mấy ngòi"],
            ["một tòa nhà có {n1} tầng", "{n2} tòa nhà có mấy tầng"],
            ["một hộp có {n1} viên bi", "{n2} hộp có mấy viên bi"]
        ]
        
        # Add more ratio templates
        objects = ["con gà", "cái bàn", "quả cam", "quả táo", "chiếc ghế", "cây bút"]
        for obj in objects:
            base_templates.append([
                f"một {obj} có {{n1}} cái",
                f"{{n2}} {obj} có mấy cái"
            ])
        
        return base_templates

    def generate_comparison_templates(self) -> List[List[str]]:
        """Generate comparison problem templates"""
        base_templates = [
            ["nam cao {n1}cm", "nam cao hơn minh {n2}cm", "hỏi minh cao bao nhiêu cm"],
            ["an nặng {n1}kg", "bình nặng hơn an {n2}kg", "hỏi bình nặng bao nhiêu kg"],
            ["lớp a có {n1} học sinh", "lớp b nhiều hơn lớp a {n2} học sinh", "hỏi lớp b có bao nhiêu học sinh"],
            ["hộp thứ nhất có {n1} viên bi", "hộp thứ hai nhiều hơn hộp thứ nhất {n2} viên", "hỏi hộp thứ hai có bao nhiêu viên"],
            ["cây thông cao {n1}m", "cây thông cao hơn cây bưởi {n2}m", "hỏi cây bưởi cao bao nhiêu mét"],
            ["lan có {n1} quả táo", "nam hơn lan {n2} quả", "hỏi nam có mấy quả"],
            ["mai có {n1} cái kẹo", "an hơn mai {n2} cái", "hỏi an có mấy cái kẹo"],
            ["bình có {n1} quyển vở", "hoa ít hơn bình {n2} quyển", "hỏi hoa có bao nhiêu quyển vở"]
        ]
        
        # Add more comparison templates
        subjects_a = ["lan", "nam", "an", "mai", "huy"]
        subjects_b = ["minh", "hoa", "bình", "linh", "thu"]
        
        for a in subjects_a:
            for b in subjects_b:
                if a != b:
                    base_templates.append([
                        f"{a} có {{n1}} quả cam",
                        f"{b} nhiều hơn {a} {{n2}} quả",
                        f"hỏi {b} có bao nhiêu quả cam"
                    ])
        
        return base_templates

    def generate_division_templates(self) -> List[List[str]]:
        """Generate division problem templates"""
        base_templates = [
            ["có {n1} quả táo", "chia đều cho {n2} bạn", "hỏi mỗi bạn được mấy quả táo"],
            ["có {n1} cái kẹo", "chia đều cho {n2} học sinh", "hỏi mỗi học sinh được mấy cái kẹo"],
            ["có {n1} quyển vở", "chia đều cho {n2} lớp", "hỏi mỗi lớp được mấy quyển vở"],
            ["có {n1} cây bút", "chia đều cho {n2} nhóm", "hỏi mỗi nhóm được mấy cây bút"],
            ["có {n1} kg gạo", "chia đều cho {n2} gia đình", "hỏi mỗi gia đình được mấy kg gạo"]
        ]
        
        # Add more division templates
        objects = ["quả xoài", "chiếc bánh", "viên kẹo", "quyển vở", "bánh kẹo"]
        for obj in objects:
            for people in [2, 3, 4, 5]:
                base_templates.append([
                    f"có {{n1}} {obj}",
                    f"chia đều cho {people} người",
                    f"mỗi người được bao nhiêu {obj.split()[0]}"
                ])
        
        return base_templates

    def generate_mixed_templates(self) -> List[List[str]]:
        """Generate mixed problem templates"""
        base_templates = [
            ["một hộp có {n1} viên bi", "chia đều cho {n2} bạn", "mỗi bạn được bao nhiêu viên"],
            ["trong vườn có {n1} quả cam", "hái {n2} quả", "còn lại bao nhiêu quả"],
            ["cửa hàng có {n1} chai nước", "bán được {n2} chai", "hỏi còn lại bao nhiêu chai"],
            ["lớp học có {n1} học sinh", "có {n2} bạn nam", "hỏi có bao nhiêu bạn nữ"],
            ["tủ sách có {n1} quyển", "cho {n2} người mượn mỗi người 2 quyển", "hỏi còn lại bao nhiêu quyển"]
        ]
        
        # Add more mixed templates
        places = ["trên bàn", "trong giỏ", "trong rổ", "trong hộp", "trong tủ"]
        objects = ["quả cam", "quả táo", "cái kẹo", "chiếc bánh", "quyển vở"]
        
        for place in places:
            for obj in objects:
                base_templates.append([
                    f"{place} có {{n1}} {obj}",
                    f"lấy đi {{n2}} {obj.split()[0]}",
                    f"hỏi còn lại bao nhiêu {obj.split()[0]}"
                ])
        
        return base_templates

    def validate_breaks(self, text: str, breaks: List[int]) -> bool:
        """Validate that break indices are within text bounds"""
        words = text.split()
        if not breaks:
            return True
        return all(break_idx < len(words) for break_idx in breaks)

    def generate_problems(self) -> List[Dict]:
        """Main generation method with validation"""
        # Initialize templates for all problem types
        template_types = [
            self.generate_ownership_templates(),
            self.generate_ratio_templates(), 
            self.generate_comparison_templates(),
            self.generate_division_templates(),
            self.generate_mixed_templates()
        ]
        
        # Generate examples for each template type
        for template_list in template_types:
            for _ in range(30):  # Generate 30 examples for each type
                template = random.choice(template_list)
                n1 = random.randint(1, 100)
                n2 = random.randint(1, min(n1, 50))  # Ensure realistic numbers
                
                sentences = [sent.format(n1=n1, n2=n2) for sent in template]
                self.add_examples_from_sentences(sentences)
        
        # Add validation
        valid_examples = []
        for example in self.examples:
            if self.validate_breaks(example["text"], example["breaks"]):
                valid_examples.append(example)
        
        self.examples = valid_examples
        return self.examples

def create_enhanced_variants(example: Dict) -> List[Dict]:
    """Create enhanced variants with different punctuation and formatting"""
    variants = []
    text = example["text"]
    breaks = example["breaks"]
    words = text.split()
    
    # 1. Original
    variants.append(example)
    
    # 2. No punctuation
    no_punct = " ".join(words)
    variants.append({"text": no_punct, "breaks": breaks})
    
    # 3. Different comma styles
    if breaks:
        # Single comma at different positions
        for break_pos in breaks[:2]:  # Limit to first two breaks
            new_words = words.copy()
            if break_pos < len(new_words):
                new_words[break_pos] = new_words[break_pos] + ","
                variants.append({"text": " ".join(new_words), "breaks": breaks})
    
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
    final_dev_data = enhanced_train_data[split_idx:split_idx + 100]  # Limit dev size
    
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