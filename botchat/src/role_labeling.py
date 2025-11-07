from state_manager import State_Manager, Entity, Attribute
from model_loader import create_classification_model, create_separate_model, create_ner_model
from typing import List, Tuple, Optional
from tabulate import tabulate
import pandas as pd
import os , re

class RelationEntity:
    """
    Biểu diễn một thực thể ngữ nghĩa được trích ra từ câu:
      - Có thể là chủ thể (agent)
      - Hoặc một vật thể / thuộc tính (attr)
      - Có thể kèm số lượng, giá trị, đơn vị

    Đây là tầng trung gian giữa dữ liệu NER và State_Manager.
    """
    def __init__(
        self,
        agent: str | None = None,
        attr: str | None = None,
        value: float | int | None = None,
        num: float | int | None = None,
        unit: str | None = None,
    ):
        self.agent = agent
        self.attr = attr
        self.value = value
        self.num = num
        self.unit = unit
        # Danh sách các thực thể tương tự (chưa cộng dồn)
        self._similar_entities: list["RelationEntity"] = []

    def __repr__(self):
        base = []
        if self.agent:
            base.append(f"agent={self.agent}")
        if self.attr:
            base.append(f"attr={self.attr}")
        if self.value is not None:
            base.append(f"value={self.value}")
        if self.num is not None:
            base.append(f"num={self.num}")
        if self.unit:
            base.append(f"unit={self.unit}")
        if self._similar_entities:
            base.append(f"similar={len(self._similar_entities)}")
        return f"<RelationEntity {', '.join(base)}>"

    # ==============================================================
    # Hỗ trợ dữ liệu
    # ==============================================================

    def to_dict(self):
        return {
            "agent": self.agent,
            "attr": self.attr,
            "value": self.value,
            "num": self.num,
            "unit": self.unit,
            "similar_entities": [e.to_dict() for e in self._similar_entities],
        }

    def is_empty(self):
        return all(
            v is None for v in [self.agent, self.attr, self.value, self.num, self.unit]
        )

    # ==============================================================
    # Gộp thực thể giống nhau (vd: "1 quả táo + 1 quả táo")
    # ==============================================================

    def is_similar_to(self, other: "RelationEntity") -> bool:
        """
        Xem hai thực thể có 'giống nhau' không.
        Nếu có agent -> không gộp.
        Nếu không có agent -> so sánh attr + unit.
        """
        if self.agent or other.agent:
            return False
        return (self.attr == other.attr) and (self.unit == other.unit)

    def merge_if_similar(self, other: "RelationEntity") -> bool:
        """
        Nếu hai thực thể giống nhau thì thêm vào danh sách `similar_entities`.
        Không cộng dồn giá trị; chỉ ghi nhận.
        """
        if self.is_similar_to(other):
            self._similar_entities.append(other)
            return True
        return False
    
    def all_entities(self) -> list["RelationEntity"]:
        """Trả về bản thân và các thực thể tương tự."""
        return [self] + self._similar_entities

    # ==============================================================
    # Chuyển đổi sang Entity và Attribute (để lưu vào State_Manager)
    # ==============================================================

    def to_state(self) -> tuple["Entity", "Attribute"] | None:
        """
        Chuyển RelationEntity thành cặp (Entity, Attribute) sẵn sàng để lưu vào State_Manager.
        Nếu không có agent hoặc attr -> trả về None.
        """
        if not self.agent or not self.attr:
            return None

        entity = Entity(self.agent)

        # Xác định giá trị cuối cùng (ưu tiên value, sau đó num)
        value = self.value if self.value is not None else self.num
        attr = Attribute(self.attr, value, self.unit)

        return entity, attr

class TextProcessor:
    def __init__(self, model_paths):
        self.classification_model = create_classification_model(model_paths['classification'])
        self.splitter_model = create_separate_model(model_paths['sentence_splitter'])
        self.ner_model = create_ner_model(model_paths['ner'])
    
    def process_text(self, text):
        # Phân loại
        classification = self.classification_model.classify_problem(text)
        
        # Tách câu
        if classification['problem_type'] != 'basic_word' and classification['problem_type'] != 'basic':
            sentences = self.splitter_model.split_sentences(text)
        else:
            sentences = {'sentences': [text]}
        
        # Phân tích NER từng câu
        ner_results = []
        for sentence in sentences['sentences']:
            try:
                ner_result = self.ner_model.extract_entities(sentence)
                ner_results.append(ner_result)
            except Exception as e:
                print(f"Lỗi khi phân tích NER cho câu: {sentence}, lỗi: {e}")
                ner_results.append({"tokens": [], "labels": [], "entities": []})
        
        return {
            'classification': classification,
            'sentences': sentences,
            'ner_results': ner_results
        }

class NERNormalizer:
    """
    Lớp chuẩn hóa dữ liệu NER:
      - Nhận đầu vào đã được gán nhãn (token, label)
      - Chuẩn hóa các quan hệ REL về dạng logic/toán học
      - Load từ điển quan hệ từ file CSV
      - Ghép các từ cùng nhãn liền kề nhau
    """

    def __init__(self, rel_csv_path: str = "data/keywords.csv"):
        self.rel_map = self._load_rel_map(rel_csv_path)

    @staticmethod
    def _load_rel_map(csv_path: str):
        """Đọc file CSV từ khóa quan hệ"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

        df = pd.read_csv(csv_path)
        rel_map = {}

        for _, row in df.iterrows():
            key = str(row["keyword"]).strip().lower()
            val = str(row["normalized"]).strip()
            if key and val:
                rel_map[key] = val

        print(f"[NERNormalizer] ✅ Đã tải {len(rel_map)} từ khóa từ {csv_path}")
        return rel_map

    def normalize(self, ner_data, show_table=False):
        """
        Chuẩn hóa dữ liệu đã gán nhãn.
        Input: list[(token, label)]
        Output: list[(token, label, rel_std)]
        """
        # Bước 1: Ghép các token cùng nhãn liền kề
        merged_data = self._merge_consecutive_labels(ner_data)
        
        # Bước 2: Chuẩn hóa quan hệ
        normalized = []
        for tok, lbl in merged_data:
            rel_std = None
            if lbl.startswith(("B-REL","I-REL")):
                rel_std = self._normalize_relation(tok.lower())
            normalized.append((tok, lbl, rel_std))

        if show_table:
            self._print_table(normalized)

        return normalized

    def _merge_consecutive_labels(self, ner_data):
        """
        Ghép các token cùng nhãn liền kề thành một token duy nhất.
        Ví dụ: [("cho", "B-REL"), ("thêm", "I-REL")] -> [("cho thêm", "B-REL")]
        """
        if not ner_data:
            return []

        merged = []
        current_token = ner_data[0][0]
        current_label = ner_data[0][1]

        for i in range(1, len(ner_data)):
            token, label = ner_data[i]
            
            # ✅ Sửa đúng chỗ lỗi ở đây
            if (label.startswith(("I-REL", "B-REL")) and current_label.startswith(("B-REL", "I-REL"))) or \
            (label == current_label and not label.startswith(("B-", "I-"))):
                current_token += " " + token
            else:
                merged.append((current_token, current_label))
                current_token = token
                current_label = label

        merged.append((current_token, current_label))
        print(f"[NERNormalizer] 🔄 Đã ghép từ {len(ner_data)} xuống {len(merged)} token")
        return merged

    def _normalize_relation(self, word: str):
        """
        Chuẩn hóa 1 từ quan hệ (REL) về dạng: +, -, *, /, have, =, >, < ...
        """
        word = word.strip().lower()

        # Tra trong từ điển CSV
        if word in self.rel_map:
            return self.rel_map[word]

        return word  # fallback

    def _print_table(self, data):
        """In kết quả dạng bảng"""
        try:
            headers = ["Token", "Label", "Rel_std"]
            table = []
            for tok, lbl, rel_std in data:
                table.append([tok, lbl, rel_std or ""])
            print(tabulate(table, headers=headers, tablefmt="grid"))
        except ImportError:
            # Fallback nếu không có tabulate
            print("Token".ljust(20) + "Label".ljust(15) + "Rel_std")
            print("-" * 50)
            for tok, lbl, rel_std in data:
                print(f"{tok.ljust(20)}{lbl.ljust(15)}{rel_std or ''}")

class StateInitializer:
    """
    Lớp khởi tạo state từ đầu ra NER đã chuẩn hóa.
    - Tạo Entity nếu agent xuất hiện (dù không có sở hữu)
    - Chỉ gán Attribute nếu có từ sở hữu (vd: "có")
    - Gom nhóm attr độc lập (vd: "1 quả táo + 2 quả táo")
    """

    def __init__(self, state_manager: State_Manager):
        self.state_manager = state_manager
        self.pending_attrs = {}  # Gom nhóm attr độc lập

    def initialize_from_tokens(self, ner_data):
        """
        ner_data: list[(token, label, rel_std)]
        Trả về: list các dict {
            "agent": str,
            "attr": str,
            "value": float|int|None,
            "unit": str|None,
            "op": str|None
        }
        """
        known_agents = {k[0] for k in self.state_manager.get_all_states()}  # lấy danh sách agent đã có

        agents_in_sentence = set()
        attributes_data = []
        independent_data = []  # ✅ dữ liệu trả về cho tính toán độc lập
        
        current_agent = None
        current_attr = None
        current_value = None
        current_unit = None
        current_has_possession = False
        current_op = None
        
        for token, label, rel_std in ner_data:
            if label.startswith("B-AGENT"):
                agent_name = token.lower()
                agents_in_sentence.add(agent_name)
                current_agent = agent_name
                
            elif label.startswith("B-ATTRIBUTE"):
                current_attr = token.lower()
                
            elif label.startswith("B-VALUE"):
                try:
                    current_value = float(token)
                except ValueError:
                    current_value = token
                    
            elif label.startswith("B-NUM"):
                try:
                    current_value = int(token)
                except ValueError:
                    current_value = token
                    
            elif label.startswith("B-UNIT"):
                current_unit = token.lower()
                
            elif label.startswith("B-REL"):
                if rel_std == "have":
                    current_has_possession = True
                elif rel_std in ["+", "-", "*", "/"]:
                    current_op = rel_std
            
            # Khi có đủ thông tin để tạo attribute (có agent hoặc không)
            if current_attr and current_value is not None:
                if current_agent:
                    attributes_data.append({
                        "agent": current_agent,
                        "attr": current_attr,
                        "value": current_value,
                        "unit": current_unit,
                        "possession": current_has_possession
                    })
                else:
                    key = (current_attr, current_unit)
                    if key not in self.pending_attrs:
                        self.pending_attrs[key] = []
                    self.pending_attrs[key].append((current_value, current_op))
                
                # Reset
                current_attr, current_value, current_unit, current_has_possession, current_op = None, None, None, False, None

        # Bước 2: Khởi tạo agent mới
        for agent in agents_in_sentence:
            if agent not in known_agents:
                entity = Entity(agent)
                empty_attr = Attribute(None, None, None)
                self.state_manager.set_state(entity, empty_attr)
                known_agents.add(agent)
                print(f"⚪ Tạo agent '{agent}' (không có thuộc tính)")

        # Bước 3: Gán thuộc tính cho agent
        for attr_data in attributes_data:
            agent = attr_data["agent"]
            attr = attr_data["attr"]
            value = attr_data["value"]
            unit = attr_data["unit"]
            has_possession = attr_data["possession"]
            
            if agent and has_possession and attr:
                entity = Entity(agent)
                attribute = Attribute(attr, value, unit)
                self.state_manager.set_state(entity, attribute)
                print(f"✅ Gán thuộc tính cho '{agent}': {value} {unit or ''} {attr}")

        # Bước 4: Xử lý thuộc tính độc lập và gom kết quả trả ra
        if self.pending_attrs:
            independent_data = self.process_pending_attrs()

        # ✅ return luôn dữ liệu độc lập để tính toán ngoài
        return independent_data

    # ==========================================================
    # Xử lý thuộc tính độc lập (tạo entity ảo)
    # ==========================================================
    def process_pending_attrs(self):
        """Xử lý các thuộc tính độc lập (không có agent)"""
        print("\n🔧 XỬ LÝ THUỘC TÍNH ĐỘC LẬP:")
        result_data = []
        
        for (attr, unit), values_ops in self.pending_attrs.items():
            for i, (value, op) in enumerate(values_ops, 1):
                entity_name = f"_independent_{attr}_{i}"
                entity = Entity(entity_name)
                attribute = Attribute(attr, value, unit)
                self.state_manager.set_state(entity, attribute)
                
                display_value = int(value) if isinstance(value, float) and value.is_integer() else value
                print(f"  {entity_name}: {display_value} {unit or ''} {attr} (op: {op})")

                # ✅ lưu vào danh sách để return
                result_data.append({
                    "agent": entity_name,
                    "attr": attr,
                    "value": value,
                    "unit": unit,
                    "op": op
                })

        self.pending_attrs.clear()
        return result_data

    # ==========================================================
    # Hiển thị trạng thái
    # ==========================================================
    def show_all_states(self):
        print("\n📊 TRẠNG THÁI HIỆN TẠI:")
        states = self.state_manager.get_all_states()
        if not states:
            print("  (Trống)")
            return
            
        for (agent, attr_name), attr in states.items():
            if attr_name:
                print(f"  {agent}: {attr_name} = {attr.get_value()} {attr.get_unit()}")
            else:
                print(f"  {agent}: [chưa có thuộc tính]")

    #========================= xử lý bài toán =========================
class ExpressionEvaluator:
    """
    Lớp xử lý và tính toán biểu thức số học từ dữ liệu NER đã gán nhãn.
    Hỗ trợ: +, -, *, /, () theo đúng quy tắc toán học.
    """

    def __init__(self):
        self.operators = {"+", "-", "*", "/", "(", ")"}

    def build_expression(self, ner_data):
        """
        Chuyển list[(token, label, rel_std)] thành chuỗi biểu thức hợp lệ.
        """
        expr_parts = []

        for token, label, rel_std in ner_data:
            token = token.strip()

            # ✅ VALUE: thêm trực tiếp
            if label.startswith("B-VALUE"):
                expr_parts.append(token)
            # ✅ REL: kiểm tra xem có phải toán tử/ngoặc hợp lệ không
            elif label.startswith(("B-REL","I-REL")):
                if rel_std in self.operators:
                    expr_parts.append(rel_std)
                elif token in self.operators:
                    expr_parts.append(token)

        expr = " ".join(expr_parts)
        expr = re.sub(r"\s+([()+\-*/])\s+", r"\1", expr)  # xóa khoảng trắng thừa quanh toán tử/ngoặc
        print(f"🧩 Biểu thức tạo được: {expr}")
        return expr

    def _is_valid_expression(self, expression):
        """Kiểm tra xem biểu thức có hợp lệ không"""
        # Chỉ cho phép chữ số, toán tử, dấu ngoặc, khoảng trắng và dấu chấm
        if not re.match(r'^[\d+\-*/().\s]+$', expression):
            return False
        # Kiểm tra số ngoặc hợp lệ
        if expression.count("(") != expression.count(")"):
            return False
        return True

    def evaluate_expression(self, expression):
        """
        Tính toán biểu thức theo đúng thứ tự ưu tiên (hỗ trợ cả ngoặc).
        """
        try:
            if not self._is_valid_expression(expression):
                raise ValueError("Biểu thức không hợp lệ hoặc sai định dạng!")

            # Dùng eval an toàn (tắt builtins)
            result = eval(expression, {"__builtins__": None}, {})
            return result

        except ZeroDivisionError:
            print("⚠️ Lỗi: chia cho 0")
            return None
        except Exception as e:
            print(f"⚠️ Lỗi khi tính toán: {e}")
            return None

    def process(self, ner_data):
        """
        Quy trình đầy đủ: tạo biểu thức và tính kết quả.
        """
        expr = self.build_expression(ner_data)
        result = self.evaluate_expression(expr)
        if result is not None:
            print(f"✅ Kết quả: {result}")
        else:
            print("❌ Không thể tính toán.")
        return result
    
class IndependentEvaluator:
    """
    Tính toán các thuộc tính độc lập được tạo trong StateInitializer.
    Hỗ trợ +, -, *, / theo đúng quy tắc toán học.
    ❌ Nếu có bất kỳ attr hoặc unit khác nhau -> dừng toàn bộ, báo lỗi.
    """

    def __init__(self, independent_data: list[dict]):
        """
        independent_data: danh sách dict gồm
          { 'agent', 'attr', 'value', 'unit', 'op' }
        """
        self.independent_data = independent_data

    def _build_expression(self):
        """
        Tạo biểu thức toán học từ danh sách value + op.
        Bảo toàn đúng phép toán gốc.
        """
        expr_parts = []
        for i, item in enumerate(self.independent_data):
            val = item["value"]
            op = item.get("op")

            # Ép kiểu hợp lệ (giữ nguyên dạng số)
            val_str = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)

            # Phần tử đầu tiên không có phép toán đứng trước
            if i == 0 or not op:
                expr_parts.append(val_str)
            else:
                expr_parts.append(f"{op} {val_str}")

        expr = " ".join(expr_parts)
        expr = re.sub(r"\s+", " ", expr).strip()
        return expr

    def _evaluate_expression(self, expr):
        """Tính biểu thức một cách an toàn."""
        try:
            return eval(expr, {"__builtins__": None}, {})
        except ZeroDivisionError:
            print("⚠️ Lỗi: chia cho 0")
            return None
        except Exception as e:
            print(f"⚠️ Lỗi khi tính biểu thức '{expr}': {e}")
            return None

    def process(self):
        """
        Xử lý toàn bộ các thuộc tính độc lập.
        Nếu phát hiện attr hoặc unit không đồng nhất → dừng toàn bộ và báo lỗi.
        """
        if not self.independent_data:
            print(" dữ liệu không đầy đủ để tính toán.")
            return None

        attrs = {v["attr"] for v in self.independent_data if v["attr"]}
        units = {v["unit"] for v in self.independent_data if v["unit"]}

        # ❌ Dừng nếu khác attr hoặc unit
        if len(attrs) > 1 or len(units) > 1:
            print("❌ Lỗi: Các thuộc tính độc lập không cùng loại hoặc không cùng đơn vị. Dừng xử lý.")
            return None

        attr = next(iter(attrs)) if attrs else None
        unit = next(iter(units)) if units else None

        expr = self._build_expression()
        result = self._evaluate_expression(expr)

        print("\n🧮 KẾT QUẢ TÍNH TOÁN THUỘC TÍNH ĐỘC LẬP:")
        print(f"  Biểu thức: {expr}")
        if result is not None:
            print(f"  ✅ Kết quả: {result} {unit or ''} {attr or ''}")
        else:
            print("  ❌ Không thể tính toán.")
        return f"  ✅ Kết quả: {result} {unit} {attr} "

class ActionProcessor:
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.list_agent = []  # Danh sách agent đã được đề cập

    def process_action_sentence(self, ner_data):
        # 1️⃣ Bỏ qua câu hỏi
        if any(lbl.startswith("B-QUESTION") for _, lbl, _ in ner_data):
            return self._handle_question(ner_data)

        # 2️⃣ Thu thập thông tin
        agents = [tok.lower() for tok, lbl, _ in ner_data if lbl.startswith("B-AGENT")]
        attrs = [tok.lower() for tok, lbl, _ in ner_data if lbl.startswith("B-ATTRIBUTE")]
        units = [tok.lower() for tok, lbl, _ in ner_data if lbl.startswith("B-UNIT")]
        rels = [(tok, rel_std) for tok, lbl, rel_std in ner_data
                if lbl.startswith(("B-REL","I-REL")) and rel_std in ["+", "-", "*", "/", ">", "<"]]
        values = [tok for tok, lbl, _ in ner_data if lbl.startswith(("B-VALUE"))]

        for ag in agents:
            if ag not in self.list_agent:
                self.list_agent.append(ag)

        # 3️⃣ Kiểm tra dữ liệu đầu vào
        current_attrs = self.state_manager.get_state_by_agent(self.list_agent[0])
        if not current_attrs:
            return f"Không tìm thấy dữ liệu cho agent '{self.list_agent[0]}'."

        for name, attr in current_attrs.items():
            attr_name = name
            unit = attr.get_unit()

        re_attr_name = attrs[0] if attrs else None
        re_unit = units[0] if units else None
        if not rels or not values:
            return "Không có phép toán hoặc giá trị, bỏ qua."
        

        if re_attr_name is None and re_unit is None:
            print("⚙️ Không có attr/unit trong câu, bỏ qua kiểm tra nhãn.")
        else:
            if re_attr_name and not re_unit:
                if re_attr_name != attr_name and re_attr_name != unit:
                    return f"❌ Không khớp: '{re_attr_name}' không tương thích với '{attr_name}' hay '{unit}'."
            elif re_unit and not re_attr_name:
                if re_unit != unit and re_unit != attr_name:
                    return f"❌ Không khớp: '{re_unit}' không tương thích với '{attr_name}' hay '{unit}'."
            else:
                if re_attr_name != attr_name:
                    return f"❌ Không thể cộng/trừ '{re_attr_name}' với '{attr_name}' – khác loại thuộc tính."
                if re_unit and unit and re_unit != unit:
                    return f"❌ Đơn vị không khớp: '{re_unit}' ≠ '{unit}'."

        rel_symbol = rels[0][1]
        value = float(values[-1])
        # 5️⃣ Xác định loại phép
        if rel_symbol in [">", "<"]:
            return self._handle_comparison(ner_data, rel_symbol, value)
        elif rel_symbol in ["+", "-", "*", "/"]:
            return self._handle_arithmetic(ner_data, rel_symbol, value)
        else:
            return "Phép không xác định."

    # 🧮 Xử lý phép toán cơ bản
    def _handle_arithmetic(self, ner_data, rel_symbol, value):
        if len(self.list_agent) == 1:
            target_agent = self.list_agent[0]
        else:
            # Tìm agent sau dấu REL
            rel_index = next((i for i, (_, _, rel_std) in enumerate(ner_data)
                              if rel_std in ["+", "-", "*", "/"]), None)
            before_rel_agents = [tok.lower() for tok, lbl, _ in ner_data[:rel_index]
                    if lbl.startswith("B-AGENT")]
            after_rel_agents = [tok.lower() for tok, lbl, _ in ner_data[rel_index + 1:]
                                if lbl.startswith("B-AGENT")]
            if len(before_rel_agents+after_rel_agents) == 1:
                target_agent = self.list_agent[0]
            else:
                target_agent = after_rel_agents[0]
            
        state = self.state_manager.get_state_by_agent(target_agent)
        for name, attr in state.items():
            attr_name = name
            old_value = attr.get_value()
            unit = attr.get_unit()

        # Thực hiện phép toán
        if rel_symbol == "+": 
            new_value = old_value + value
        elif rel_symbol == "-": 
            new_value = old_value - value
        elif rel_symbol == "*": 
            new_value = old_value * value
        elif rel_symbol == "/": 
            new_value = old_value / value if value != 0 else old_value
        else:
            return "Phép toán không hợp lệ."

        # Cập nhật state
        self.state_manager.update_state(Entity.find(target_agent),
                                        Attribute(attr_name, None, unit),
                                        new_value)

        if not self.list_agent or self.list_agent[-1] != target_agent:
            self.list_agent.append(target_agent)

            return f"{target_agent} hiện có {new_value} {unit or ''} {attr_name or ''}"

    # ⚖️ Xử lý phép so sánh
    def _handle_comparison(self, ner_data, rel_symbol, value):
        rel_index = next((i for i, (_, _, rel_std) in enumerate(ner_data)
                          if rel_std in [">", "<"]), None)

        before_rel_agents = [tok.lower() for tok, lbl, _ in ner_data[:rel_index]
                             if lbl.startswith("B-AGENT")]
        after_rel_agents = [tok.lower() for tok, lbl, _ in ner_data[rel_index + 1:]
                            if lbl.startswith("B-AGENT")]

        left_agent = before_rel_agents[-1] if before_rel_agents else None
        right_agent = after_rel_agents[0] if after_rel_agents else None

        left_state = self.state_manager.get_state_by_agent(left_agent) if left_agent else None
        right_state = self.state_manager.get_state_by_agent(right_agent) if right_agent else None

        left_val, right_val = None, None
        attr_name, unit = None, None

        if left_state:
            for name, attr in left_state.items():
                left_val = attr.get_value()
                attr_name = name
                unit = attr.get_unit()
        if right_state:
            for name, attr in right_state.items():
                right_val = attr.get_value()
                attr_name = attr_name or name
                unit = unit or attr.get_unit()

        # Logic xử lý
        if rel_symbol == "<":
            if left_val is not None and right_val is None:
                new_value = left_val + value
                target_agent = right_agent
            elif right_val is not None and left_val is None:
                new_value = right_val - value
                target_agent = left_agent
            else:
                return f"So sánh: {left_agent} ({left_val}) < {right_agent} ({right_val}) + {value}"

        elif rel_symbol == ">":
            if left_val is not None and right_val is None:
                new_value = left_val - value
                target_agent = right_agent
            elif right_val is not None and left_val is None:
                new_value = right_val + value
                target_agent = left_agent
            else:
                return f"So sánh: {left_agent} ({left_val}) > {right_agent} ({right_val}) + {value}"

        # Cập nhật state
        self.state_manager.update_state(Entity.find(target_agent),
                                        Attribute(attr_name, None, unit),
                                        new_value)
        if not self.list_agent or self.list_agent[-1] != target_agent:
            self.list_agent.append(target_agent)

            return f"{target_agent} hiện có {new_value} {unit or ''} {attr_name or ''}"
        
    def _handle_question(self, ner_data):
        # Lấy agent
        agents = [tok.lower() for tok, lbl, _ in ner_data if lbl.startswith("B-AGENT")]
        attrs = [tok.lower() for tok, lbl, _ in ner_data if lbl.startswith("B-ATTRIBUTE")]
        units = [tok.lower() for tok, lbl, _ in ner_data if lbl.startswith("B-UNIT")]

        if not agents:
            return "❓ Không tìm thấy agent trong câu hỏi."

        target_agent = agents[0]
        state = self.state_manager.get_state_by_agent(target_agent)
        if not state:
            return f"❌ Không có dữ liệu cho agent '{target_agent}'."

        matched_results = []

        for name, attr in state.items():
            attr_name = name
            unit = attr.get_unit()
            value = attr.get_value()
        print(f"Debug: Kiểm tra {attr_name}, giá trị: {value}, đơn vị: {unit}")
        # Trường hợp có cả attr và unit trong câu hỏi
        if attrs and units:
            if attrs[0] == attr_name and units[0] == unit:
                matched_results.append((attr_name, value, unit))
        # Chỉ có attr
        elif attrs and not units:
            if attrs[0] == attr_name or attrs[0] == unit:
                matched_results.append((attr_name, value, unit))
        # Chỉ có unit
        elif units and not attrs:
            if units[0] == unit or units[0] == attr_name:
                matched_results.append((attr_name, value, unit))

        # Không có attr/unit cụ thể => trả toàn bộ
        elif not attrs and not units:
            return "❓ Câu hỏi không rõ ràng, vui lòng cung cấp thêm thông tin."
            
        if not matched_results:
            return f"❌ Không tìm thấy dữ liệu tương ứng cho '{target_agent}'."

        # Ghép kết quả trả về
        answer_lines = [f"{target_agent} có {val} {u or ''} {a}" for a, val, u in matched_results]
        return " | ".join(answer_lines)




def redirect(result):
    normalizer = NERNormalizer("data/keywords.csv")
    final_result = None
    state_manager = State_Manager()
    initializer = StateInitializer(state_manager)
    for ner in result.get('ner_results', []):
        problem_type = result.get('classification', {}).get('problem_type')

    if problem_type == 'basic':
        for ner in result.get('ner_results', []):
            normalized = normalizer.normalize(ner['entities'])
            print(normalized)
            final_result = ExpressionEvaluator().process(normalized)
        return final_result
    
    elif problem_type == 'basic_word':
        for ner in result.get('ner_results', []):
            normalized = normalizer.normalize(ner['entities'])
            print(normalized)           
            independent_data = initializer.initialize_from_tokens(normalized)
            final_result = IndependentEvaluator(independent_data).process()
        return final_result
    
    elif problem_type == 'comparison' or problem_type == 'ownership':
        andler = ActionProcessor(state_manager)
        for ner in result.get('ner_results', []):
            entities = ner.get("entities", [])
            normalized = normalizer.normalize(entities, show_table=False)
            print(normalized)
            initializer.initialize_from_tokens(normalized)
            final_result=andler.process_action_sentence(normalized)
        return final_result



def main():
    model_paths = {
        'classification': 'src/model/classification_model',
        'sentence_splitter': 'src/model/sent_split_model', 
        'ner': 'src/model/ner_model'
    }
    processor = TextProcessor(model_paths)
    result = processor.process_text("lan có 30 quả táo, nam hơn lan 10 quả, hỏi nam có mấy quả ?")
    print(redirect(result))
    
if __name__ == "__main__":
    main()