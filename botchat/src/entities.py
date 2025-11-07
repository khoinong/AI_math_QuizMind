#lớp Entity chứa danh từ,chủ ngữ, chủ sở hữu trong câu.
class Entity:
    _all_entities = []  # Danh sách toàn bộ entity đã được tạo

    def __init__(self, agent: str):
        self._agent = agent
        # Tự động lưu vào danh sách khi khởi tạo
        Entity._all_entities.append(self)       
    def set_agent(self,agent):
        self._agent= agent

    def get_agent(self):
        return self._agent
    @classmethod
    def find(cls, name: str):
        """
        Tìm Entity theo tên (chuỗi). 
        Trả về Entity nếu tìm thấy, None nếu không.
        """
        for e in cls._all_entities:
            if e._agent.lower() == name.lower():
                return e
        return None
    
    def __repr__(self):
        return f"Entity({self._agent})"
    
# lớp attribute thể hiện đặc tính của 1 vật thể vd: táo,cam ... hoặc trạng thái của chủ thể vd: đang đi,đang chạy
class Attribute :
    def __init__(self,name_attribute,value,unit):
        self._name_attribute= name_attribute
        self._value= value
        self._unit= unit

    def __repr__(self):
        return f"Attribute({self._name_attribute}, {self._value}, {self._unit})"

    def set_name_attribute(self,name_attribute):
        self._name_attribute=name_attribute
    def set_value(self,value):
        self._value= value
    def set_unit(sefl,unit):
        sefl._unit=unit
    
    def get_name_attribute(self):
        return self._name_attribute
    def get_value(self):
        return self._value
    def get_unit(self):
        return self._unit

def main():
    agent1 = Entity("lan")
    print(agent1.get_agent())

if __name__ == "__main__":
    main()