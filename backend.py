import os
import json
import tempfile
import time
import subprocess
import sys
from datetime import datetime, timedelta

import chromadb
import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
import ollama
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv
import easyocr
import fitz  # PyMuPDF

app = Flask(__name__)
CORS(app)
load_dotenv()

# ========== تحميل النماذج ==========
print("⏳ جاري تحميل نموذج التضمين (Embedding)...")
model_embedding = SentenceTransformer('intfloat/multilingual-e5-small')

print("⏳ جاري تهيئة EasyOCR للغة العربية...")
try:
    easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
except Exception as e:
    print(f"⚠️ خطأ في تهيئة EasyOCR: {e}")
    easyocr_reader = None

# ========== الاتصال بقاعدة البيانات ==========
client_db = chromadb.PersistentClient(path="legal_db")
try:
    collection = client_db.get_collection(name="algerian_law")
except NotFoundError:
    print("⚠️ المجموعة غير موجودة، سيتم إنشاؤها...")
    collection = client_db.create_collection(name="algerian_law")

# ========== سجل الملفات المضافة ==========
HISTORY_FILE = "processed_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(history), f, ensure_ascii=False, indent=2)


# ========== دوال تنظيف النص العربي ==========
def clean_arabic_text(text):
    if not text:
        return ""
    try:
        reshaped = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped)
        return bidi_text
    except Exception as e:
        print(f"⚠️ خطأ في تنظيف النص: {e}")
        return text


def has_substantial_text(text):
    return text and len(text.strip()) > 50


# ========== استخراج النص من PDF/صور ==========
def extract_text_with_fallback(file_path, file_extension=None):
    full_text = ""
    method_used = None

    if file_extension == '.pdf' or (file_extension is None and file_path.lower().endswith('.pdf')):
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"
            if has_substantial_text(full_text):
                method_used = "pdfplumber (نصوص رقمية)"
                return full_text, method_used
        except Exception as e:
            print(f"⚠️ فشل pdfplumber: {e}")

        if easyocr_reader:
            try:
                doc = fitz.open(file_path)
                ocr_text = ""
                for page_num in range(len(doc)):
                    pix = doc.load_page(page_num).get_pixmap(dpi=150)
                    img_path = f"temp_page_{page_num}.png"
                    pix.save(img_path)
                    result = easyocr_reader.readtext(img_path, detail=0, paragraph=True)
                    if result:
                        ocr_text += " ".join(result) + "\n"
                    os.remove(img_path)
                doc.close()
                if has_substantial_text(ocr_text):
                    return ocr_text, "EasyOCR (PDF ممسوح)"
            except Exception as e:
                print(f"⚠️ فشل EasyOCR: {e}")

    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        if easyocr_reader:
            result = easyocr_reader.readtext(file_path, detail=0, paragraph=True)
            if result:
                return " ".join(result), "EasyOCR (صورة)"

    return "", None


def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    buffer = []
    current_len = 0
    for word in words:
        buffer.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(buffer))
            buffer = []
            current_len = 0
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def add_file_to_library(file_path, original_filename=None):
    display_name = original_filename or os.path.basename(file_path)
    history = load_history()
    if display_name in history:
        return f"⚠️ الملف '{display_name}' تمت إضافته مسبقاً."

    full_text, method = extract_text_with_fallback(file_path, os.path.path.splitext(file_path)[1].lower())
    if not full_text:
        return "❌ تعذر استخراج النص."

    clean_text = clean_arabic_text(full_text)
    chunks = chunk_text(clean_text)

    embeddings = [model_embedding.encode("passage: " + c).tolist() for c in chunks]
    ids = [f"{display_name}_{idx}" for idx in range(len(chunks))]
    metadatas = [{"source": display_name} for _ in chunks]

    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    history.add(display_name)
    save_history(history)
    return f"✅ تمت إضافة '{display_name}' بنجاح عبر {method}."


def initial_scan_and_build():
    if collection.count() > 0:
        return
    folders = ["data/01--- قوانين وزارة التجارة", "data/التجارة الالكترونية", "data/قوانين السجل التجاري",
               "data/contrats_exemples", "data/كتب تجارية"]
    for folder in folders:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        print(add_file_to_library(os.path.join(root, file)))


# ========== دالة الإجابة باستخدام Ollama ==========
def ask_lawyer(query):
    try:
        query_embedding = model_embedding.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)

        context = "\n".join(results['documents'][0]) if results['documents'] else "لا يوجد سياق قانوني متاح."

        full_prompt = f"""أنت مستشار قانوني جزائري محترف.
مهمتك: تقديم إجابات قانونية دقيقة ومنظمة بناءً فقط على النصوص القانونية المرفقة.

قواعد التنسيق الإلزامية:
- استخدم عناوين رئيسية على شكل: 1-العنوان (استخدم الأرقام)
- استخدم عناوين فرعية على شكل: أ-العنوان(استخدم الحرف)
- افصل بين كل عنوان و عنوان بسطر فارغ.
- استعمل خط كبير للعنوانين و خط رقيق للاجابة 
- لا تنسخ النص حرفياً من المصادر، بل أعد صياغته بلغة قانونية واضحة ومختصرة.
- لا تذكر عبارات مثل "بناءً على النصوص أعلاه" أو "وفقاً للمصادر". ابدأ الإجابة مباشرة.

النصوص القانونية:
{context}

سؤال المستخدم:
{query}

الإجابة (باللغة العربية):"""

        response = ollama.chat(model='glm-5:cloud', messages=[{'role': 'user', 'content': full_prompt}])
        return {"answer": response['message']['content']}
    except Exception as e:
        print(f"❌ خطأ في Ollama: {e}")
        return {"answer": "حدث خطأ أثناء محاولة معالجة السؤال محلياً. تأكد من تشغيل برنامج Ollama."}


# ========== مولد العقود ==========
def generate_contract(contract_type, parties, subject, duration, amount):
    prompt = f"""أنت مستشار قانوني جزائري متخصص في صياغة العقود.
بناءً على النصوص القانونية الجزائرية (قانون التجارة، قانون الصفقات العمومية، القانون المدني)، قم بإنشاء عقد كامل من نوع "{contract_type}" يتضمن المواد التالية على الأقل:
- تعريف الأطراف
- موضوع العقد
- المدة: {duration}
- المبلغ: {amount}
- التزامات الطرفين
- شروط الدفع
- الجزاءات والغرامات التأخيرية
- الضمانات
- تسوية النزاعات (التحكيم أو المحاكم الجزائرية)
- أحكام عامة (القوة القاهرة، اللغة، عدد النسخ)

أطراف العقد:
{parties}

موضوع العقد:
{subject}

اكتب العقد بلغة قانونية واضحة، مرقماً المواد (مادة 1، مادة 2...)، منسقاً بأسطر فارغة بين المواد. لا تذكر أي جمل تمهيدية مثل "بناءً على طلبك". ابدأ مباشرة بنص العقد.
"""
    response = ollama.chat(model='kimi-k2.5:cloud', messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content']


@app.route('/generate_contract', methods=['POST'])
def api_generate_contract():
    data = request.get_json()
    contract_type = data.get('type')
    parties = data.get('parties')
    subject = data.get('subject')
    duration = data.get('duration')
    amount = data.get('amount')
    if not all([contract_type, parties, subject]):
        return jsonify({"error": "يرجى ملء الحقول المطلوبة"}), 400
    contract_text = generate_contract(contract_type, parties, subject, duration, amount)
    return jsonify({"contract": contract_text})


@app.route('/download_contract_pdf', methods=['POST'])
def download_contract_pdf():
    data = request.get_json()
    contract_text = data.get('contract')
    if not contract_text:
        return jsonify({"error": "لا يوجد نص عقد"}), 400
    try:
        from fpdf import FPDF
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
        from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', size=12)
    for line in contract_text.split('\n'):
        pdf.multi_cell(0, 10, line)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    return send_file(temp_file.name, as_attachment=True, download_name='contrat_genere.pdf')


# ========== محلل المستندات ==========
@app.route('/analyze_document', methods=['POST'])
def analyze_document():
    if 'file' not in request.files:
        return jsonify({"error": "لا يوجد ملف"}), 400
    file = request.files['file']
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        text, method = extract_text_with_fallback(tmp.name, ext)
    os.unlink(tmp.name)
    if not text:
        return jsonify({"error": "تعذر استخراج النص من الملف"}), 400
    prompt = f"""أنت خبير قانوني جزائري. قم بتحليل النص التالي وأخرج:
1. ملخص (3-5 جمل)
2. نقاط الخطر القانونية (Clauses à risque) - إن وجدت، وإلا اذكر "لا توجد نقاط خطر واضحة"
3. توصيات عملية للمستخدم

النص:
{text[:4000]}

أجب بالتنسيق التالي:
**الملخص:**
...
**نقاط الخطر:**
- ...
**التوصيات:**
- ...
"""
    response = ollama.chat(model='kimi-k2.5:cloud', messages=[{'role': 'user', 'content': prompt}])
    return jsonify({"analysis": response['message']['content']})


# ========== حاسبة المواعيد القانونية ==========
LEGAL_DEADLINES = {
    "تقادم دعوى مدنية": 15,
    "تقادم دعوى تجارية": 10,
    "الطعن في صفقة عمومية (بعد التبليغ)": 60,
    "الطعن في قرار إداري": 30,
    "إنهاء عقد عمل (إشعار مسبق)": 30,
}


@app.route('/calculate_deadlines', methods=['POST'])
def calculate_deadlines():
    data = request.get_json()
    action = data.get('action')
    start_date_str = data.get('start_date')
    if not action or not start_date_str:
        return jsonify({"error": "يرجى تحديد الإجراء والتاريخ"}), 400
    if action not in LEGAL_DEADLINES:
        return jsonify({"error": "نوع الإجراء غير معروف"}), 400
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    days = LEGAL_DEADLINES[action]
    delta = timedelta(days=days)
    end_date = start_date + delta
    return jsonify({
        "action": action,
        "start_date": start_date_str,
        "deadline_date": end_date.strftime('%Y-%m-%d'),
        "days_remaining": max(0, (end_date - datetime.now()).days),
        "legal_basis": "المادة المرجعية حسب القانون الجزائري"
    })


# ========== مسارات API ==========
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')


@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "الرجاء إدخال سؤال"}), 400
    return jsonify(ask_lawyer(query))


@app.route('/api/ask', methods=['POST'])
def api_ask():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "الرجاء إدخال سؤال"}), 400
    return jsonify(ask_lawyer(query))


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "لا يوجد ملف"}), 400
    file = request.files['file']
    file_extension = os.path.splitext(file.filename).filename.lower()  # تم تصحيحها لاسم الملف
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        file.save(tmp.name)
        result = add_file_to_library(tmp.name, file.filename)
    os.unlink(tmp.name)
    return jsonify({"message": result})


@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({"chunks_count": collection.count()})


if __name__ == '__main__':
    os.makedirs("data/contrats_exemples", exist_ok=True)
    initial_scan_and_build()
    app.run(host='0.0.0.0', port=5000, debug=False)