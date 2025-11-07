# src/app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import os,sys
sys.path.append("src")
from role_labeling import TextProcessor, redirect
# ứng dụng Flask
app = Flask(__name__, template_folder="templates", static_folder="templates/public")

# Static route: (tuỳ nếu bạn cần)
@app.route('/public/<path:filename>')
def serve_public(filename):
    public_dir = os.path.join(app.root_path, "templates/public")
    return send_from_directory(public_dir, filename)

# ====== Load models ONCE at startup ======
MODEL_PATHS = {
    'classification': 'src/model/classification_model',
    'sentence_splitter': 'src/model/sent_split_model',
    'ner': 'src/model/ner_model'
}

print("⏳ Loading models (may take a while)...")
processor = TextProcessor(MODEL_PATHS)
print("✅ Models loaded")

# ====== API ======
@app.route('/')
def home():
    # index.html ở templates/index.html
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "empty input"}), 400

    # xử lý
    result = processor.process_text(text)
    final = redirect(result)   # redirect trả final_result (hoặc None)
    # trả kiểu JSON, final có thể là số, str, None
    return jsonify({"reply": final})

if __name__ == "__main__":
    # chỉ dùng để dev local; production dùng gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
