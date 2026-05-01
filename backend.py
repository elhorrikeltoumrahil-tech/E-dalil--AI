# ====================================================================
# backend.py - الخادم الخلفي للمنصة القانونية E-dalil DZ
# تم التعديل لاستخدام NVIDIA API (DeepSeek-V4-Pro) بدلاً من DeepSeek المباشر
# ====================================================================

# ---------------------- استيراد المكتبات الأساسية ----------------------
import os
import json
import tempfile
import time
import subprocess
import sys
from datetime import datetime, timedelta

# ---------------------- مكتبات معالجة النصوص والذكاء الاصطناعي ----------------------
import chromadb
import pdfplumber
import arabic_reshaper
from bidi.algorithm import get_display
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
import requests
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI      # استخدم OpenAI client للاتصال بـ NVIDIA
import easyocr
import fitz

# ---------------------- تهيئة تطبيق Flask ----------------------
app = Flask(__name__)
CORS(app)
load_dotenv()  # تحميل المتغيرات من .env

# ---------------------- إعدادات NVIDIA API ----------------------
# استخدام المفتاح من متغير البيئة NVIDIA_API_KEY
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    print("⚠️ تحذير: لم يتم العثور على NVIDIA_API_KEY في ملف .env")
    print("⚠️ سيتم استخدام وضع المحاكاة (mock) للإجابات.")

# تهيئة عميل OpenAI مع عنوان NVIDIA
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY if NVIDIA_API_KEY else "dummy-key"
)

# اسم النموذج المعتمد من NVIDIA (DeepSeek-V4-Pro)
NVIDIA_MODEL = "deepseek-ai/deepseek-v4-pro"

# ---------------------- تحميل نماذج الذكاء الاصطناعي المحلية ----------------------
print("⏳ جاري تحميل نموذج التضمين (Embedding)...")
model_embedding = SentenceTransformer('intfloat/multilingual-e5-small')

print("⏳ جاري تهيئة EasyOCR للغة العربية...")
try:
    easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
except Exception as e:
    print(f"⚠️ خطأ في تهيئة EasyOCR: {e}")
    easyocr_reader = None

# ---------------------- الاتصال بقاعدة بيانات المتجهات ChromaDB ----------------------
client_db = chromadb.PersistentClient(path="legal_db")
try:
    collection = client_db.get_collection(name="algerian_law")
except NotFoundError:
    print("⚠️ المجموعة غير موجودة، سيتم إنشاؤها...")
    collection = client_db.create_collection(name="algerian_law")

# ---------------------- إدارة سجل الملفات المضافة ----------------------
HISTORY_FILE = "processed_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(history), f, ensure_ascii=False, indent=2)

# ---------------------- دوال تنظيف النص العربي ----------------------
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

# ---------------------- استخراج النص من PDF أو الصور ----------------------
def extract_text_with_fallback(file_path, file_extension=None):
    full_text = ""
    method_used = None

    if file_extension == '.pdf' or (file_extension is None and file_path.lower().endswith('.pdf')):
        # pdfplumber للنصوص الرقمية
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

        # EasyOCR للـ PDF الممسوحة
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

# ---------------------- إضافة ملف إلى المكتبة ----------------------
def add_file_to_library(file_path, original_filename=None):
    display_name = original_filename or os.path.basename(file_path)
    history = load_history()
    if display_name in history:
        return f"⚠️ الملف '{display_name}' تمت إضافته مسبقاً."

    full_text, method = extract_text_with_fallback(file_path, os.path.splitext(file_path)[1].lower())
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

# ---------------------- فحص المجلدات الافتراضية عند بدء التشغيل ----------------------
def initial_scan_and_build():
    if collection.count() > 0:
        return
    folders = ["data/01--- قوانين وزارة التجارة", "data/التجارة الالكترونية", "data/قوانين السجل التجاري", "data/contrats_exemples", "data/كتب تجارية"]
    for folder in folders:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        print(add_file_to_library(os.path.join(root, file)))

# ---------------------- دالة استدعاء NVIDIA API (بديل DeepSeek) ----------------------
def deepseek_chat(prompt: str, system_message: str = "أنت مستشار قانوني جزائري محترف.") -> str:
    """
    إرسال طلب إلى NVIDIA API (DeepSeek-V4-Pro) باستخدام عميل OpenAI.
    تعيد النص الناتج، أو رسالة خطأ في حالة الفشل.
    """
    if not NVIDIA_API_KEY:
        return "⚠️ لم يتم تكوين مفتاح NVIDIA API. يرجى إضافة NVIDIA_API_KEY في ملف .env."

    try:
        # استدعاء النموذج مع دعم رسالة النظام
        completion = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,       # دقة عالية للإجابات القانونية
            top_p=0.95,
            max_tokens=2000,
            stream=False           # نفضل الحصول على الرد كاملاً
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"❌ خطأ في طلب NVIDIA API: {e}")
        return f"حدث خطأ أثناء الاتصال بخدمة NVIDIA: {str(e)}"

# ---------------------- الإجابة على الأسئلة باستخدام RAG ----------------------
# 1. استبدل دالة ask_lawyer ودالة التضمين بهذا الكود المحسن
def get_embedding_api(text):
    """جلب التضمين (Embedding) من NVIDIA API بدلاً من تشغيله محلياً"""
    try:
        response = client.embeddings.create(
            input=[text],
            model="nvidia/nv-embedqa-e5-v5"  # نموذج تضمين خفيف وسريع من NVIDIA
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in embedding: {e}")
        return None


# 2. تحديث دالة البحث لتستخدم الـ API
def ask_lawyer(query):
    try:
        # استخدام الـ API للتضمين بدلاً من النموذج المحلي model_embedding
        query_embedding = get_embedding_api("query: " + query)
        if not query_embedding:
            return {"answer": "خطأ في جلب التضمين من السحاب."}

        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        context = "\n".join(results['documents'][0]) if results['documents'] else "لا يوجد سياق."

        full_prompt = f"بناءً على النصوص: {context}\nالسؤال: {query}"

        # استدعاء الشات (تأكد من ضبط stream=True في دالة deepseek_chat إذا أردت السرعة)
        answer = deepseek_chat(full_prompt)
        return {"answer": answer}
    except Exception as e:
        return {"answer": f"حدث خطأ: {str(e)}"}


# 3. حل مشكلة الـ PDF (دعم اللغة العربية)
@app.route('/download_contract_pdf', methods=['POST'])
def download_contract_pdf():
    data = request.get_json()
    contract_text = data.get('contract')

    from fpdf import FPDF
    # ملاحظة: يجب تحميل خط Amiri-Regular.ttf ووضعه في مجلد المشروع
    pdf = FPDF()
    pdf.add_page()

    # تحميل خط يدعم العربية (يجب أن يكون الملف موجوداً بجانب backend.py)
    try:
        pdf.add_font('Amiri', '', 'Amiri-Regular.ttf')
        pdf.set_font('Amiri', size=12)
    except:
        pdf.set_font('Arial', size=12)  # احتياطي

    for line in contract_text.split('\n'):
        # إعادة تشكيل النص ليدعم الـ RTL
        reshaped_text = arabic_reshaper.reshape(line)
        bidi_text = get_display(reshaped_text)
        pdf.multi_cell(0, 10, bidi_text, align='R')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf.output(temp_file.name)
    return send_file(temp_file.name, as_attachment=True, download_name='contract_dz.pdf')

# ---------------------- مولد العقود ----------------------
def generate_contract(contract_type, parties, subject, duration, amount):
    prompt = f"""قم بإنشاء عقد قانوني جزائري من نوع "{contract_type}" يتضمن المواد التالية:
- تعريف الأطراف (الأطراف: {parties})
- موضوع العقد (الموضوع: {subject})
- المدة: {duration if duration else "غير محددة"}
- المبلغ: {amount if amount else "يُحدد لاحقاً"}
- التزامات الطرفين
- شروط الدفع
- الجزاءات والغرامات التأخيرية
- الضمانات
- تسوية النزاعات (التحكيم أو المحاكم الجزائرية)
- أحكام عامة (القوة القاهرة، اللغة، عدد النسخ)

اكتب العقد بلغة قانونية واضحة، مرقماً المواد (مادة 1، مادة 2، ...)، مع ترك سطر فارغ بين المواد. لا تذكر أي جمل تمهيدية، ابدأ مباشرة بنص العقد."""
    system_msg = "أنت مستشار قانوني جزائري متخصص في صياغة العقود. استخدم اللغة العربية الفصحى واتبع أحكام القانون الجزائري."
    return deepseek_chat(prompt, system_message=system_msg)

# ---------------------- مسارات API ----------------------

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

    prompt = f"""قم بتحليل النص القانوني التالي (المستخرج من مستند مرفوع) وأخرج:
1. ملخص (3-5 جمل)
2. نقاط الخطر القانونية - إن وجدت، وإلا اذكر "لا توجد نقاط خطر واضحة"
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
    system_msg = "أنت خبير قانوني جزائري في تحليل العقود والوثائق القانونية."
    analysis = deepseek_chat(prompt, system_message=system_msg)
    return jsonify({"analysis": analysis})

# ---------------------- حاسبة المواعيد القانونية ----------------------
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

# مسار إضافي (اختياري) للتوافق مع الكود المقدم من المستخدم
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
    file_extension = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        file.save(tmp.name)
        result = add_file_to_library(tmp.name, file.filename)
    os.unlink(tmp.name)
    return jsonify({"message": result})

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({"chunks_count": collection.count()})

# ---------------------- تشغيل الخادم ----------------------
if __name__ == '__main__':
    os.makedirs("data/contrats_exemples", exist_ok=True)
    initial_scan_and_build()
    app.run(host='0.0.0.0', port=5000, debug=False)