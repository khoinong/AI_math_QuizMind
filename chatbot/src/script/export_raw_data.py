import json
import random

def generate_math_problems(num_samples=300):
    samples = []
    addition_templates = [
        "{person1} có {a} {item}. {person2} có {b} {item}. Hỏi cả hai có bao nhiêu {item}?",
        "Một {place} có {a} {item1} và {b} {item2}. Hỏi cửa hàng có bao nhiêu {item1} và {item2}?",
        "Tổng số {item} của {person1} và {person2} là bao nhiêu? Biết {person1} có {a} {item} và {person2} có {b} {item}."
    ]
    subtraction_templates = [
        "{person} có {a} {item}, {pronoun} cho bạn {b} {item}. Hỏi {person} còn lại bao nhiêu {item}?",
        "Lúc đầu có {a} {item}, sau đó mất {b} {item}. Hỏi còn lại bao nhiêu {item}?",
        "{person} có {a} {item}, {pronoun} dùng hết {b} {item}. Hỏi còn lại bao nhiêu {item}?"
    ]
    
    contexts = {
        "items": ["quả táo", "quả cam", "cái kẹo", "quyển sách", "cái bút", "viên bi",
                 "chiếc áo", "chiếc quần", "đôi giày", "cái mũ", "cái cặp", "quả bóng","cái ghế","cái bàn","quả dưa hấu","quả chuối"],
        "items1": ["áo sơ mi", "quần", "giày", "mũ", "váy", "áo khoác"],
        "items2": ["quần", "áo", "tất", "váy", "nón", "găng tay"],
        "places": ["cửa hàng", "nhà kho", "tủ quần áo", "siêu thị", "thư viện"],
        "persons": ["Lan", "Minh", "Hoa", "Nam", "An", "Bình", "Tùng", "Chi"],
        "persons1": ["Lan", "Minh", "Hoa", "An"],
        "persons2": ["Minh", "Hoa", "Nam", "Bình", "Tùng", "Chi","Bảo"],
        "pronouns": ["cô ấy", "anh ấy", "bạn ấy"]
    }
    
    for _ in range(num_samples):
        if random.choice([True, False]):  # Addition
            template = random.choice(addition_templates)
            a = random.randint(1, 500)
            b = random.randint(1, 500)
            equation = f"{a} + {b}"
            answer = a + b
            
            if template == addition_templates[0]:
                context = {
                    "person1": random.choice(contexts["persons1"]),
                    "person2": random.choice(contexts["persons2"]),
                    "item": random.choice(contexts["items"]),
                    "a": a,
                    "b": b
                }
            elif template == addition_templates[1]:
                context = {
                    "place": random.choice(contexts["places"]),
                    "item1": random.choice(contexts["items1"]),
                    "item2": random.choice(contexts["items2"]),
                    "a": a,
                    "b": b
                }
            else:
                context = {
                    "person1": random.choice(contexts["persons1"]),
                    "person2": random.choice(contexts["persons2"]),
                    "item": random.choice(contexts["items"]),
                    "a": a,
                    "b": b
                }
            
            problem_text = template.format(**context)
            
        else:  # Subtraction
            template = random.choice(subtraction_templates)
            a = random.randint(2, 500)
            b = random.randint(1, a-1)
            equation = f"{a} - {b}"
            answer = a - b
            
            context = {
                "person": random.choice(contexts["persons"]),
                "item": random.choice(contexts["items"]),
                "a": a,
                "b": b,
                "pronoun": random.choice(contexts["pronouns"])
            }
            
            problem_text = template.format(**context)
        
        samples.append({
            "problem_text": problem_text,
            "number": [a, b],
            "equation": equation,
            "answer": answer,
            "target_formula": f"{equation} = {answer}"
        })
    
    return samples

# Generate samples
samples = generate_math_problems(1000)

# Save to JSONL
out_path = "chatbot/data/formula_dataset.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for sample in samples:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"✅ Đã tạo {len(samples)} mẫu dữ liệu và lưu vào {out_path}")
