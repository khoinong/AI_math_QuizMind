from entities import Entity,Attribute

class State_Manager:
    def __init__(self):
        self._states = {}

    def set_state(self, entity: Entity, attribute: Attribute):
        """
        Lưu nguyên attribute (gồm name, value, unit).
        Mỗi entity + name_attribute chỉ giữ được 1 attribute (ghi đè nếu có).
        """
        key = (entity.get_agent(), attribute.get_name_attribute())
        self._states[key] = attribute

    def update_state(self, entity: Entity, attribute: Attribute, new_value):
        """
        Cập nhật giá trị attribute (value).
        Giữ nguyên unit, name.
        """
        key = (entity.get_agent(), attribute.get_name_attribute())
        if key in self._states:
            # Update existing attribute with new value
            existing_attr = self._states[key]
            existing_attr.set_value(new_value)
        else:
            # Create new attribute with the value
            attribute.set_value(new_value)
            self._states[key] = attribute

    def get_state(self, entity: Entity, attribute: Attribute):
        """Truy vấn: trả về object Attribute hoặc None"""
        key = (entity.get_agent(), attribute.get_name_attribute())
        return self._states.get(key, None)
    
    def get_state_by_agent(self, agent_name: str):
        """
        ✅ Truy vấn tất cả trạng thái (Attribute) thuộc về 1 agent.
        Trả về dict: {attribute_name: Attribute object}
        """
        result = {}
        for (agent, attr_name), attr in self._states.items():
            if agent.lower() == agent_name.lower():
                result[attr_name] = attr
        return result if result else None
    
    def get_all_states(self):
        return self._states


if __name__ == "__main__":
    lan = Entity("Lan")
    minh = Entity("Minh")

    # Create attributes with initial values
    tao = Attribute("táo", 5, "quả")
    cam = Attribute("cam", 3, "quả")

    sm = State_Manager()
    sm.set_state(lan, tao)   # Lan có 5 quả táo
    sm.set_state(minh, cam)  # Minh có 3 quả cam
    sm.update_state(lan, tao, 7)  # Lan cập nhật số táo thành 7 quả táo

    # Test the results
    lan_tao = sm.get_state(lan, tao)
    minh_cam = sm.get_state(minh, cam)
    
    print(f"Lan có: {lan_tao.get_value()} {lan_tao.get_unit()} {lan_tao.get_name_attribute()}")   # -> 7 quả táo
    print(f"Minh có: {minh_cam.get_value()} {minh_cam.get_unit()} {minh_cam.get_name_attribute()}") # -> 3 quả cam
    
    # Print all states
    # print("Tất cả trạng thái:")
    # for key, attr in sm.get_all_states().items():
    #     agent, attr_name = key
    #     print(f"{agent}: {attr.get_value()} {attr.get_unit()} {attr_name}")
    lan_states = sm.get_state_by_agent("lan")

    if lan_states:
        for attr_name, attr in lan_states.items():
            print(f"Lan có {attr.get_value()} {attr.get_unit()} {attr_name}")
    else:
        print("Không tìm thấy dữ liệu cho agent này.")