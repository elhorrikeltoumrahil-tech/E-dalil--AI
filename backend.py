import os
import json
import re
import tempfile
import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fitz  # PyMuPDF
from paddleocr import PaddleOCR
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# ========== إعدادات متغيرات البيئة ==========
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# ========== تحميل نموذج التضمين ==========
print("⏳ جاري تحميل نموذج التضمين...")
model_embedding = SentenceTransformer('intfloat/multilingual-e5-small')

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

# ========== معالج المستندات الذكي (SmartDocumentProcessor) ==========
class SmartDocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.full_text = ""
        # تهيئة PaddleOCR للغة العربية
        self.ocr = PaddleOCR(lang='ar')

        # تعبير نمطي للبحث عن عناوين المواد والفصول والأقسام (يدعم الأرقام العربية والإنجليزية والرومانية)
        self.heading_pattern = re.compile(
            r'(المادة|الفصل|القسم|الفرع|الباب|المبحث)\s+(\d+|الأول|الثاني|الثالث|الرابع|الخامس|[IVXLCDM]+)',
            re.IGNORECASE
        )

    def fix_arabic_rtl(self, text):
        """
        إصلاح النص العربي المقلوب (RTL) الذي قد يظهر معكوساً بسبب PaddleOCR.
        تعمل هذه الدالة على عكس ترتيب الكلمات في السطر إذا كان النص يبدو معكوساً.
        """
        if not text:
            return ""
        lines = text.split('\n')
        fixed_lines = []
        for line in lines:
            # إذا كان السطر يحتوي على عربية ويبدو مقلوباً (وجود كلمة "ةداملا" مثلاً)
            if re.search(r'[\u0600-\u06FF]', line) and ("ةداملا" in line or "نوناق" in line):
                fixed_line = line[::-1]  # عكس السطر بالكامل
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def extract_text(self):
        """استخراج النص من ملف PDF باستخدام PyMuPDF للنصوص العادية و PaddleOCR للصفحات الممسوحة"""
        doc = fitz.open(self.pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                # إذا كانت الصفحة تحتوي على نص قابل للاستخراج مباشرة
                self.full_text += page_text + "\n"
            else:
                # إذا كانت الصفحة فارغة (ممسوحة ضوئياً)، استخدم PaddleOCR
                print(f"   📸 الصفحة {page_num+1} ممسوحة ضوئياً، جاري استخدام OCR...")
                pix = page.get_pixmap()
                img_path = f"temp_page_{page_num}.png"
                pix.save(img_path)
                try:
                    result = self.ocr.ocr(img_path, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            self.full_text += line[1][0] + " "
                except Exception as e:
                    print(f"   ⚠️ خطأ في OCR للصفحة {page_num+1}: {e}")
                finally:
                    if os.path.exists(img_path):
                        os.remove(img_path)
        # إصلاح اتجاه النص العربي (RTL)
        self.full_text = self.fix_arabic_rtl(self.full_text)
        return self.full_text

    def smart_chunk(self, text):
        """تقطيع النص بناءً على العناوين (المواد، الفصول، إلخ) مع الاحتفاظ بالسياق"""
        chunks = []
        current_chunk = ""
        lines = text.split('\n')
        for line in lines:
            # إذا وجدنا سطراً يبدو كعنوان قانوني، نبدأ قطعة جديدة
            if self.heading_pattern.search(line):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        # إذا لم يتم العثور على أي عنوان، قم بتقطيع النص إلى أجزاء بطول 500 كلمة كحل احتياطي
        if len(chunks) <= 1 and len(text.split()) > 500:
            words = text.split()
            buffer = []
            current_len = 0
            for word in words:
                buffer.append(word)
                current_len += len(word) + 1
                if current_len >= 500:
                    chunks.append(" ".join(buffer))
                    buffer = []
                    current_len = 0
            if buffer:
                chunks.append(" ".join(buffer))
        return chunks

    def process(self):
        """تنفيذ العملية الكاملة: استخراج + تقطيع"""
        print(f"   📄 معالجة الملف: {os.path.basename(self.pdf_path)}")
        full_text = self.extract_text()
        if not full_text.strip():
            return []
        chunks = self.smart_chunk(full_text)
        print(f"   ✅ تم استخراج {len(chunks)} قطعة نصية.")
        return chunks

# ========== دالة إضافة ملف PDF إلى المكتبة (معدلة) ==========
def add_pdf_to_library(pdf_path):
    rel_path = os.path.basename(pdf_path)
    history = load_history()
    if rel_path in history:
        return f"⚠️ الملف '{rel_path}' تمت إضافته مسبقاً."

    try:
        # استخدام المعالج الذكي
        processor = SmartDocumentProcessor(pdf_path)
        chunks = processor.process()
        if not chunks:
            return "❌ لم يتم استخراج أي نصوص من الملف."

        # حساب المتجهات والإضافة إلى قاعدة البيانات
        embeddings = []
        ids = []
        metadatas = []
        for idx, chunk in enumerate(chunks):
            vector = model_embedding.encode("passage: " + chunk).tolist()
            embeddings.append(vector)
            ids.append(f"{rel_path}_part{idx+1}")
            metadatas.append({"source": rel_path})

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        history.add(rel_path)
        save_history(history)
        return f"✅ تمت إضافة {len(chunks)} جزء من الملف '{rel_path}' بنجاح."
    except Exception as e:
        return f"❌ خطأ: {str(e)}"

# ========== دالة المسح الأولي (إذا كانت قاعدة البيانات فارغة) ==========
def initial_scan_and_build():
    if collection.count() > 0:
        print("✅ قاعدة البيانات تحتوي بالفعل على بيانات. تخطي المسح الأولي.")
        return

    print("📂 قاعدة البيانات فارغة. بدء المسح الأولي للمجلدات...")
    # تحديد مجلدات PDF الموجودة (يمكنك تعديل المسارات حسب هيكلك)
    pdf_folders = [
        "data/قوانين السجل التجاري",
        "data/التجارة الالكترونية",
        "data/01--- قوانين وزارة التجارة"
    ]
    all_pdfs = []
    for folder in pdf_folders:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        all_pdfs.append(os.path.join(root, file))

    if not all_pdfs:
        print("⚠️ لم يتم العثور على أي ملفات PDF في المجلدات المحددة.")
        return

    print(f"📄 تم العثور على {len(all_pdfs)} ملف PDF. جاري المعالجة...")
    for pdf_path in all_pdfs:
        print(f"   معالجة: {pdf_path}")
        result = add_pdf_to_library(pdf_path)
        print(f"   {result}")
    print("🎉 انتهى المسح الأولي بنجاح.")

    for folder in pdf_folders:
        print(f"🔍 جاري فحص المجلد: {folder}")
        if os.path.exists(folder):
            print(f"   ✅ المجلد موجود")
        else:
            print(f"   ❌ المجلد غير موجود")
# ========== دوال البحث والإجابة (نفس السابق) ==========
def ask_lawyer(query):
    if not client:
        return {"answer": "❌ مفتاح Gemini API غير مضبوط. يرجى تعيين GEMINI_API_KEY.", "sources": []}

    query_vector = model_embedding.encode("query: " + query).tolist()
    try:
        results = collection.query(query_embeddings=[query_vector], n_results=5)
    except Exception as e:
        return {"answer": f"⚠️ خطأ في البحث: {e}", "sources": []}

    if not results['documents'][0]:
        return {"answer": "عذراً، لم أتمكن من العثور على معلومات متعلقة بسؤالك.", "sources": []}

    context = "\n\n".join(results['documents'][0])

    system_prompt = """أنت مستشار قانوني جزائري خبير.
    مهمتك هي الإجابة على أسئلة المستخدمين بناءً على النصوص القانونية المقدمة فقط.

    تعليمات مهمة:
    - قدم إجابة شاملة وكاملة دون اختصار.
    - اذكر المصدر باختصار (مثل: "المادة 5 من القانون التجاري").
    - استخدم اللغة العربية الفصحى الواضحة.
    - إذا لم تجد المعلومة في النصوص، أخبر المستخدم بذلك بوضوح."""

    user_prompt = f"النصوص القانونية المتوفرة:\n{context}\n\nالسؤال: {query}"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=4096,
                top_p=0.9
            )
        )
        sources = []
        for i, doc in enumerate(results['documents'][0]):
            src = results['metadatas'][0][i]['source']
            sources.append({"source": src, "text_preview": doc[:300] + "..."})
        return {"answer": response.text, "sources": sources}
    except Exception as e:
        return {"answer": f"⚠️ خطأ في الاتصال: {str(e)}", "sources": []}

# ========== مسارات API ==========
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "الرجاء إدخال سؤال"}), 400
    result = ask_lawyer(query)
    return jsonify(result)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "لم يتم رفع أي ملف"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "اسم الملف فارغ"}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "الرجاء رفع ملف PDF فقط"}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        file.save(tmp.name)
        result = add_pdf_to_library(tmp.name)
    os.unlink(tmp.name)
    return jsonify({"message": result})

@app.route('/stats', methods=['GET'])
def stats():
    count = collection.count()
    return jsonify({"chunks_count": count})

# ========== تشغيل الخادم ==========
if __name__ == '__main__':
    initial_scan_and_build()
    app.run(host='0.0.0.0', port=5000, debug=False)