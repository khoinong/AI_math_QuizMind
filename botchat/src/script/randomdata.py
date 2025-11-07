import random

# Các thành phần từ vựng
agentss = ["tổng","hiệu","tích","thương"]
relations1 =["là","bằng","="]
relations2 =["cộng","trừ","nhân","chia","+","-","*","/"]
relations3 =["lớn","nhỏ","to","bé","kém","thấp","cao"]
relations4 =["<",">","bằng","="]
agents = ["Lan", "Minh", "An", "Bình", "Hoa", "Nam", "Hà", "Tuấn", "Hùng", "Vy", "Quân", "Phương", "Dũng", "Thảo", "Trang", "Long"]
relations = ["có", "giữ", "đưa", "lấy", "nhận", "mua", "bán", "cho", "tặng", "cộng", "thêm","của","và","mượn","trừ","nhân","chia","bằng","cùng"]
units = ["cái", "chiếc", "bộ", "tấm", "quả","con", "bình", "chai", "hộp", "túi", "gói", "kg", "lít", "đôi", "cặp","cm","m","km","giờ","phút","giây","viên","hòn"]
attributes = ["táo", "cam", "chuối", "nho", "dưa", "laptop", "sách", "bút", "vở", "bánh", "sữa", "cơm", "phở",
              "trà","khế", "xoài", "dừa","rau", "thịt", "cá", "gà", "vịt", "heo", "bò", "xe", "quần", "áo", "giày", "mũ", "mèo","chim","chó","bom","pháo","súng","bóng","đá","bóng","rổ","bóng","chuyền","bóng","đá","bóng","bàn","bóng","bầu","cầu","lông","quần","vợt",
              "tay", "chân", "mắt", "tai", "mũi", "miệng", "đầu", "cổ"]
question_words = ["mấy", "Hỏi"]

# Hàm sinh một câu dữ liệu
def generate_sample():
    agent2 = random.choice(agentss)
    agent = random.choice(agents)
    agentsa = random.choice(agents)
    rel = random.choice(relations)
    rel1 = random.choice(relations1)
    rel2 = random.choice(relations2)
    rel3 = random.choice(relations3)
    rel4 = random.choice(relations4)
    question_word = random.choice(question_words)

    val = random.randint(1,20)
    unit = random.choice(units)
    attr = random.choice(attributes)
    
    sample = [
        (agent, "B-AGENT"),
        (rel, "B-REL"),
        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
        ("và", "B-REL"),
        (random.randint(1,100), "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
    ]
    
    sample2 = [
        (agent2, "B-AGENT"),
        ("của", "B-REL"),
        (agent, "B-AGENT"),
        ("và", "B-REL"),
        (agentsa, "B-AGENT"),
        (rel1, "B-REL"),
        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
    ]
    
    sample3 = [
        (val, "B-VALUE"),
        (attr, "B-ATTRIBUTE"),
        (rel2, "B-REL"),
        (random.randint(1,100), "B-VALUE"),
        (attr, "B-ATTRIBUTE"),
    ]

    sample4 = [
        (agent, "B-AGENT"),
        ("hơn", "I-REL"),    
        (agentsa, "B-AGENT"),
        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
    ]
    
    sample5 = [
        (agent, "B-AGENT"),
        (rel4, "B-REL"),    
        (agentsa, "B-AGENT"),
        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
    ]

    # Mẫu câu hỏi: 5 con gà có mấy cái chân
    sample6 = [
        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
        ("có", "B-REL"),
        ("mấy", "B-QUESTION"),
        ("cái", "I-QUESTION"),
        (attr, "B-ATTRIBUTE"),
    ]

    # Mẫu câu hỏi: số cam của nam hơn lan là 3 quả
    sample7 = [
        ("số", "B-AGENT"),
        (attr, "B-ATTRIBUTE"),
        ("của", "B-REL"),
        (agent, "B-AGENT"),
        ("hơn", "B-REL"),
        (agentsa, "B-AGENT"),
        ("là", "B-REL"),
        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
    ]

    # Mẫu câu hỏi: Hỏi nam có bao nhiêu quả cam
    sample8 = [
        ("Hỏi", "B-QUESTION"),
        (agent, "B-AGENT"),
        ("có", "B-REL"),
        ("bao", "B-QUESTION"),
        ("nhiêu", "I-QUESTION"),
        (attr, "B-ATTRIBUTE"),
    ]

    sample9 = [
        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),

        (rel4, "B-REL"),

        (val, "B-VALUE"),
        (unit, "B-UNIT"),
        (attr, "B-ATTRIBUTE"),
        ("=", "B-REL"),
        ("bao", "B-QUESTION"),
        ("nhiêu", "I-QUESTION"),
        (unit, "B-UNIT"),
    ]

    base = [
        (val, "B-VALUE"),
        (rel2, "B-REL"),
        ("(", "I-REL"),
        (val, "B-VALUE"),
        (rel2, "B-REL"),
        (val, "B-VALUE"),
        (")", "I-REL"),
        (rel1, "B-REL"),
    ]


    return [base]

# Sinh nhiều mẫu và lưu file .conll
with open("synthetic_data.conll", "w", encoding="utf-8") as f:
    for _ in range(500):  # số mẫu cần sinh
        samples = generate_sample()
        for sample in samples:
            for token, label in sample:
                f.write(f"{token} {label}\n")
            f.write("\n")  # ngắt câu