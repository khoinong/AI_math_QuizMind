# Chatbot NLP System

## Description
Dự án này là một hệ thống chatbot xử lý ngôn ngữ tự nhiên dựa trên mô hình phân loại câu, tách câu và NER (Named Entity Recognition). Hệ thống giúp hiểu các câu liên quan đến số lượng, sở hữu, so sánh, quyền sở hữu và các thao tác trên đối tượng trong không gian dữ liệu.

Một phần quan trọng của hệ thống là nhận diện các thành phần như:
- Entity: chủ thể hoặc người tham gia
- Value: số lượng
- Unit: đơn vị đo lường
- Attribute: đối tượng, vật phẩm hoặc thuộc tính
- REL: hành động hoặc quan hệ giữa các thực thể

Thông qua quy trình này, chatbot có thể chuyển câu người dùng thành logic xử lý, cập nhật trạng thái và trả ra kết quả phù hợp.

## Features
- Phân loại loại câu đầu vào như basic, basic_word, comparison, ownership, ratio
- Tách câu phức tạp thành nhiều câu nhỏ để xử lý riêng lẻ
- Trích xuất thực thể bằng mô hình NER
- Nhận diện hành động và quan hệ giữa các thực thể
- Xử lý phép tính cơ bản và biểu thức số học
- Cập nhật trạng thái dữ liệu qua State Manager
- Hỗ trợ trả lời theo dạng câu hỏi và câu lệnh
- Giao diện web đơn giản bằng Flask

## Tech Stack
- Python
- Flask
- PyTorch
- TensorFlow / Keras
- Transformers
- scikit-learn
- NLTK
- spaCy
- pandas
- NumPy
- Matplotlib
- Seaborn
- JSON Lines

## Architecture
Hệ thống được xây dựng theo mô hình xử lý dữ liệu theo từng tầng:

1. Input Layer
   - Người dùng gửi câu văn bản qua giao diện web hoặc API.

2. Preprocessing Layer
   - Tách câu
   - Chuẩn hóa văn bản
   - Phân loại bài toán

3. NER Layer
   - Trích xuất Entity, Value, Unit, Attribute, REL

4. Reasoning Layer
   - Xác định hành động và quan hệ
   - Tính toán biểu thức hoặc cập nhật trạng thái

5. State Management Layer
   - Lưu trữ và cập nhật trạng thái của các thực thể

6. Output Layer
   - Trả kết quả qua API JSON hoặc hiển thị trên giao diện web

## Installation
1. Clone dự án:
   git clone https://github.com/khoinong/AI_math_QuizMind.git

2. Vào thư mục dự án:
   cd chatbot

3. Tạo môi trường ảo (khuyến nghị):
   python -m venv venv
   source venv/bin/activate
   # trên Windows: venv\Scripts\activate

4. Cài đặt dependencies:
   pip install -r requirements.txt

5. Kiểm tra các thư viện cần thiết và đảm bảo môi trường Python đã được kích hoạt.

## How to run
1. Chạy ứng dụng Flask:
   python app.py

2. Mở trình duyệt tại địa chỉ:
   http://localhost:5000

3. Nhập câu cần xử lý vào giao diện, ví dụ:
   - Lan có 3 quả táo,Minh cho Lan 2 quả cam. hỏi lan có mấy quả ?
   - 5 + 3 * 2
   - 1 quả táo + 3 quả táo

4. Hệ thống sẽ xử lý và trả về kết quả thông qua API hoặc giao diện web.

---

Project status: prototype / research chatbot with NLP processing and state-based reasoning.
