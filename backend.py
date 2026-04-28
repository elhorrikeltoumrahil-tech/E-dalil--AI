# ====================================================================
# backend.py - الخادم الخلفي للمنصة القانونية E-dalil DZ
# ====================================================================

# ---------------------- استيراد المكتبات الأساسية ----------------------
import os                     # للتعامل مع نظام الملفات: إنشاء مجلدات، حذف ملفات، فحص وجود مسارات
import json                   # لقراءة وكتابة ملفات JSON (مثل سجل الملفات المضافة)
import tempfile               # لإنشاء ملفات مؤقتة بأسماء عشوائية (لحفظ الملفات المرفوعة مؤقتًا)
import time                   # (احتياطي) يمكن استخدامه لقياس وقت التنفيذ أو التأخير
import subprocess             # لتشغيل أوامر خارجية من داخل Python (مثل تثبيت مكتبة fpdf2)
import sys                    # للوصول إلى مترجم Python الحالي (مثل sys.executable)
from datetime import datetime, timedelta  # للتعامل مع التواريخ: حساب الفروق، إضافة أيام، تنسيق

# ---------------------- مكتبات معالجة النصوص والذكاء الاصطناعي ----------------------
import chromadb               # قاعدة بيانات متجهات (Vector DB) مفتوحة المصدر لتخزين واسترجاع المقاطع النصية
import pdfplumber             # لاستخراج النصوص من ملفات PDF الرقمية (غير الممسوحة)
import arabic_reshaper        # لإعادة تشكيل الحروف العربية التي تظهر منفصلة (لتصلح للعرض)
from bidi.algorithm import get_display  # لمعالجة اتجاه النص العربي (RTL) والأرقام الإنجليزية
from chromadb.errors import NotFoundError  # خطأ خاص بـ chromadb عند عدم وجود مجموعة (collection)
from sentence_transformers import SentenceTransformer  # تحميل نماذج تحويل النص إلى متجهات (embeddings)

import requests               # لإرسال طلبات HTTP إلى DeepSeek API (بديل عن ollama)
from flask import Flask, request, jsonify, send_from_directory, send_file  # إطار Flask
from flask_cors import CORS   # للسماح بمشاركة الموارد عبر النطاقات (CORS) - ضروري للواجهة الأمامية
from dotenv import load_dotenv # لقراءة المتغيرات البيئية من ملف .env (مثل مفتاح API)

import easyocr                # مكتبة OCR لاستخراج النصوص من الصور أو PDF الممسوحة ضوئيًا (تدعم العربية)
import fitz                   # PyMuPDF - لتحويل صفحات PDF إلى صور (تمهيدًا لاستخدام OCR)

# ---------------------- تهيئة تطبيق Flask ----------------------
app = Flask(__name__)         # إنشاء كائن التطبيق الرئيسي
CORS(app)                     # تفعيل CORS للسماح بطلبات من أي نطاق (للواجهة الأمامية)
load_dotenv()                 # تحميل المتغيرات من ملف .env إلى بيئة التشغيل (os.environ)

# ---------------------- إعدادات DeepSeek API (تقرأ من .env) ----------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")          # مفتاح API السري (يبدأ بـ sk-)
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")  # رابط API
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # اسم النموذج المستخدم

# تحذير إذا لم يتم العثور على المفتاح
if not DEEPSEEK_API_KEY:
    print("⚠️ تحذير: لم يتم العثور على DEEPSEEK_API_KEY في ملف .env")
    print("⚠️ سيتم استخدام وضع المحاكاة (mock) للإجابات.")

# ---------------------- تحميل نماذج الذكاء الاصطناعي المحلية ----------------------
print("⏳ جاري تحميل نموذج التضمين (Embedding)...")
model_embedding = SentenceTransformer('intfloat/multilingual-e5-small')  # نموذج متعدد اللغات يحول النص إلى متجه 384-بعد

print("⏳ جاري تهيئة EasyOCR للغة العربية...")
try:
    # إنشاء قارئ OCR يدعم العربية والإنجليزية، يعمل على CPU، بدون رسائل تفصيلية
    easyocr_reader = easyocr.Reader(['ar', 'en'], gpu=False, verbose=False)
except Exception as e:
    print(f"⚠️ خطأ في تهيئة EasyOCR: {e}")
    easyocr_reader = None      # إذا فشل، نضع القيمة None وسنتجنب استخدامه لاحقًا

# ---------------------- الاتصال بقاعدة بيانات المتجهات ChromaDB ----------------------
client_db = chromadb.PersistentClient(path="legal_db")    # تخزين مستمر في مجلد legal_db
try:
    collection = client_db.get_collection(name="algerian_law")  # محاولة الحصول على مجموعة باسم "algerian_law"
except NotFoundError:
    print("⚠️ المجموعة غير موجودة، سيتم إنشاؤها...")
    collection = client_db.create_collection(name="algerian_law")  # إنشاء مجموعة جديدة إذا لم تكن موجودة

# ---------------------- إدارة سجل الملفات المضافة (لعدم تكرار الإضافة) ----------------------
HISTORY_FILE = "processed_history.json"   # اسم الملف الذي يحتفظ بأسماء الملفات المضافة سابقًا

def load_history():
    """تحميل قائمة الملفات التي سبق إضافتها إلى المكتبة من ملف JSON."""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))    # تحويل القائمة إلى مجموعة (set) لتسهيل البحث ومنع التكرار
    return set()   # إذا لم يكن الملف موجودًا، نعيد مجموعة فارغة

def save_history(history):
    """حفظ قائمة الملفات المضافة (كمجموعة) إلى ملف JSON."""
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(history), f, ensure_ascii=False, indent=2)  # تحويل set إلى list قبل الحفظ

# ---------------------- دوال تنظيف النص العربي للعرض ----------------------
def clean_arabic_text(text):
    """
    تقوم بإعادة تشكيل النص العربي (وصل الحروف المنفصلة) ثم إعادة توجيه النص
    ليكون قابلاً للعرض من اليمين إلى اليسار (RTL).
    """
    if not text:
        return ""
    try:
        reshaped = arabic_reshaper.reshape(text)   # إعادة تشكيل الحروف العربية
        bidi_text = get_display(reshaped)          # ضبط الاتجاه (RTL)
        return bidi_text
    except Exception as e:
        print(f"⚠️ خطأ في تنظيف النص: {e}")
        return text   # في حالة الفشل، نعيد النص الأصلي

def has_substantial_text(text):
    """تتحقق مما إذا كان النص يحتوي على محتوى ذي معنى (أكثر من 50 حرفًا بعد إزالة المسافات)."""
    return text and len(text.strip()) > 50

# ---------------------- استخراج النص من PDF أو الصور (مع خيارات احتياطية) ----------------------
def extract_text_with_fallback(file_path, file_extension=None):
    """
    تحاول استخراج النص من ملف PDF أو صورة باستخدام عدة طرق:
    1. pdfplumber للنصوص الرقمية في PDF.
    2. EasyOCR (إذا كان PDF ممسوحًا ضوئيًا أو صورة).
    تعيد (النص المستخرج, طريقة الاستخراج) أو ("", None) إذا فشل كل شيء.
    """
    full_text = ""
    method_used = None

    # ----- الحالة: ملف PDF -----
    if file_extension == '.pdf' or (file_extension is None and file_path.lower().endswith('.pdf')):
        # المحاولة الأولى: pdfplumber لاستخراج النصوص الرقمية
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

        # المحاولة الثانية: EasyOCR (لـ PDF الممسوح ضوئيًا)
        if easyocr_reader:
            try:
                doc = fitz.open(file_path)           # فتح PDF باستخدام PyMuPDF
                ocr_text = ""
                for page_num in range(len(doc)):
                    # تحويل الصفحة إلى صورة PNG بدقة 150 نقطة في البوصة
                    pix = doc.load_page(page_num).get_pixmap(dpi=150)
                    img_path = f"temp_page_{page_num}.png"
                    pix.save(img_path)                # حفظ الصورة المؤقتة
                    # استخراج النص من الصورة باستخدام EasyOCR (بدون تفاصيل، مع تجميع الفقرات)
                    result = easyocr_reader.readtext(img_path, detail=0, paragraph=True)
                    if result:
                        ocr_text += " ".join(result) + "\n"
                    os.remove(img_path)               # حذف الصورة المؤقتة بعد استخدامها
                doc.close()
                if has_substantial_text(ocr_text):
                    return ocr_text, "EasyOCR (PDF ممسوح)"
            except Exception as e:
                print(f"⚠️ فشل EasyOCR: {e}")

    # ----- الحالة: ملف صورة -----
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        if easyocr_reader:
            result = easyocr_reader.readtext(file_path, detail=0, paragraph=True)
            if result:
                return " ".join(result), "EasyOCR (صورة)"

    return "", None   # فشل كل المحاولات

def chunk_text(text, chunk_size=500):
    """
    تقسيم النص الطويل إلى أجزاء صغيرة (chunks) كل جزء بحجم تقريبي chunk_size (عدد الأحرف).
    تستخدم لتخزين النصوص في قاعدة المتجهات والحصول على نتائج بحث دقيقة.
    """
    words = text.split()
    chunks = []
    buffer = []
    current_len = 0
    for word in words:
        buffer.append(word)
        current_len += len(word) + 1      # +1 للمسافة بعد الكلمة
        if current_len >= chunk_size:     # إذا تجاوزنا الحجم المطلوب
            chunks.append(" ".join(buffer))
            buffer = []
            current_len = 0
    if buffer:                             # إضافة آخر جزء متبقي
        chunks.append(" ".join(buffer))
    return chunks

# ---------------------- إضافة ملف إلى مكتبة المتجهات ----------------------
def add_file_to_library(file_path, original_filename=None):
    """
    دالة رئيسية لإضافة ملف (PDF أو صورة) إلى قاعدة المتجهات ChromaDB.
    تتضمن الخطوات: التحقق من عدم التكرار، استخراج النص، التنظيف، التقسيم،
    توليد التضمينات (embeddings)، التخزين، وتحديث السجل.
    """
    display_name = original_filename or os.path.basename(file_path)   # اسم الملف للعرض
    history = load_history()
    if display_name in history:
        return f"⚠️ الملف '{display_name}' تمت إضافته مسبقاً."

    # استخراج النص باستخدام الطرق المتعددة
    full_text, method = extract_text_with_fallback(file_path, os.path.splitext(file_path)[1].lower())
    if not full_text:
        return "❌ تعذر استخراج النص."

    clean_text = clean_arabic_text(full_text)   # تنظيف النص العربي
    chunks = chunk_text(clean_text)             # تقسيم إلى أجزاء

    # توليد التضمينات: نضيف بادئة "passage: " لأن نموذج e5-small يتوقع هذا التمييز
    embeddings = [model_embedding.encode("passage: " + c).tolist() for c in chunks]
    ids = [f"{display_name}_{idx}" for idx in range(len(chunks))]    # معرف فريد لكل جزء
    metadatas = [{"source": display_name} for _ in chunks]           # بيانات وصفية: مصدر الملف

    # إضافة إلى مجموعة ChromaDB
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

    # تحديث السجل وحفظه
    history.add(display_name)
    save_history(history)

    return f"✅ تمت إضافة '{display_name}' بنجاح عبر {method}."

# ---------------------- فحص المجلدات المحددة وإضافة الملفات تلقائيًا عند بدء التشغيل ----------------------
def initial_scan_and_build():
    """
    إذا كانت المكتبة فارغة، تفحص مجموعة من المجلدات المحددة مسبقًا وتضيف جميع ملفات PDF الموجودة فيها.
    تُستخدم لبناء المكتبة أول مرة.
    """
    if collection.count() > 0:   # إذا كانت المكتبة تحتوي بالفعل على بيانات، نخرج
        return
    # قائمة المجلدات التي تحتوي على النصوص القانونية
    folders = ["data/01--- قوانين وزارة التجارة", "data/التجارة الالكترونية", "data/قوانين السجل التجاري", "data/contrats_exemples", "data/كتب تجارية"]
    for folder in folders:
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):     # تجول في المجلد وجميع المجلدات الفرعية
                for file in files:
                    if file.lower().endswith('.pdf'):  # فقط ملفات PDF
                        print(add_file_to_library(os.path.join(root, file)))

# ---------------------- دالة استدعاء DeepSeek API ----------------------
def deepseek_chat(prompt: str, system_message: str = "أنت مستشار قانوني جزائري محترف.") -> str:
    """
    إرسال طلب إلى DeepSeek API وإرجاع النص الناتج.
    في حال عدم توفر مفتاح API، تعيد رسالة خطأ (وضع المحاكاة).
    """
    if not DEEPSEEK_API_KEY:
        # وضع المحاكاة - نعيد ردًا افتراضيًا
        return "⚠️ لم يتم تكوين مفتاح DeepSeek API. يرجى إضافة DEEPSEEK_API_KEY في ملف .env. (هذه رسالة تجريبية)"

    # رؤوس الطلب (headers)
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",   # المصادقة باستخدام Bearer token
        "Content-Type": "application/json"
    }

    # جسم الطلب (payload)
    payload = {
        "model": DEEPSEEK_MODEL,                         # النموذج المستخدم
        "messages": [
            {"role": "system", "content": system_message},  # تحديد دور المساعد وشخصيته
            {"role": "user", "content": prompt}             # سؤال المستخدم
        ],
        "temperature": 0.3,      # درجة العشوائية (0 = حرفي، 1 = إبداعي)؛ 0.3 يعطي إجابات دقيقة
        "max_tokens": 2000,      # الحد الأقصى لعدد الرموز (tokens) في الرد
        "stream": False          # نريد الرد كاملاً وليس على شكل تدفق (stream)
    }

    try:
        # إرسال الطلب إلى DeepSeek API مع مهلة 60 ثانية
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()          # يرفع استثناء إذا كان كود HTTP خطأ (4xx أو 5xx)
        data = response.json()               # تحويل الاستجابة (JSON) إلى قاموس Python
        # استخراج نص الرد من الهيكل المعتاد (مشابه لـ OpenAI API)
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"❌ خطأ في طلب DeepSeek API: {e}")
        return f"حدث خطأ أثناء الاتصال بخدمة DeepSeek: {str(e)}"
    except (KeyError, IndexError) as e:
        print(f"❌ خطأ في تحليل استجابة DeepSeek: {e}")
        return "حدث خطأ في معالجة الرد من الخدمة."

# ---------------------- دالة الإجابة على الأسئلة القانونية ----------------------
def ask_lawyer(query):
    """
    تبحث في قاعدة المتجهات عن السياقات القانونية ذات الصلة بسؤال المستخدم،
    ثم تطلب من DeepSeek توليد إجابة دقيقة ومنظمة بناءً على تلك السياقات فقط.
    """
    try:
        # تحويل سؤال المستخدم إلى تضمين (embedding) مع بادئة "query: "
        query_embedding = model_embedding.encode(["query: " + query]).tolist()
        # استرجاع أقرب 3 مقاطع نصية (سياقات) من قاعدة المتجهات
        results = collection.query(query_embeddings=query_embedding, n_results=3)

        # دمج السياقات المسترجعة في نص واحد، مع التعامل مع حالة عدم وجود نتائج
        context = "\n".join(results['documents'][0]) if results['documents'] else "لا يوجد سياق قانوني متاح."

        # بناء التعليمات (prompt) التي سترسل إلى النموذج
        full_prompt = f"""بناءً على النصوص القانونية التالية فقط، أجب عن سؤال المستخدم بشكل دقيق ومنظم.

النصوص القانونية:
{context}

سؤال المستخدم:
{query}

تعليمات التنسيق:
- استخدم عناوين رئيسية مرقمة مثل "1-العنوان"
- استخدم عناوين فرعية مثل "أ-العنوان"
- افصل بين الأقسام بسطر فارغ
- لا تنسخ النص حرفياً، بل أعد صياغته
- ابدأ الإجابة مباشرة دون مقدمات

الإجابة:"""
        system_msg = "أنت مستشار قانوني جزائري محترف. مهمتك تقديم إجابات دقيقة بناءً فقط على النصوص القانونية المتاحة."

        answer = deepseek_chat(full_prompt, system_message=system_msg)
        return {"answer": answer}
    except Exception as e:
        print(f"❌ خطأ في ask_lawyer: {e}")
        return {"answer": "حدث خطأ أثناء معالجة سؤالك. يرجى المحاولة لاحقاً."}

# ---------------------- مولد العقود باستخدام DeepSeek ----------------------
def generate_contract(contract_type, parties, subject, duration, amount):
    """
    توليد عقد قانوني كامل بناءً على البيانات المدخلة (نوع العقد، الأطراف، الموضوع، المدة، المبلغ).
    يستخدم DeepSeek API لإنشاء نص منظم (مواد مرقمة، لغة قانونية).
    """
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

# ---------------------- مسارات API (واجهات الخادم) ----------------------

@app.route('/generate_contract', methods=['POST'])
def api_generate_contract():
    """واجهة API لتوليد العقد: تستقبل JSON يحتوي على نوع العقد، الأطراف، الموضوع، المدة، المبلغ."""
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
    """واجهة API لتحويل نص العقد إلى ملف PDF وتحميله للمستخدم."""
    data = request.get_json()
    contract_text = data.get('contract')
    if not contract_text:
        return jsonify({"error": "لا يوجد نص عقد"}), 400
    # محاولة استيراد fpdf، وإذا لم تكن مثبتة نقوم بتثبيتها تلقائيًا
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
    """
    واجهة API لتحليل مستند مرفوع (PDF أو صورة):
    تستخرج النص، ترسله إلى DeepSeek مع برومبت تحليل، وتعيد الملخص ونقاط الخطر والتوصيات.
    """
    if 'file' not in request.files:
        return jsonify({"error": "لا يوجد ملف"}), 400
    file = request.files['file']
    ext = os.path.splitext(file.filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        file.save(tmp.name)
        text, method = extract_text_with_fallback(tmp.name, ext)
    os.unlink(tmp.name)   # حذف الملف المؤقت بعد استخدامه
    if not text:
        return jsonify({"error": "تعذر استخراج النص من الملف"}), 400

    # بناء برومبت التحليل (يحدد المخرجات المطلوبة)
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
# قاموس يربط كل إجراء قانوني بعدد الأيام المحددة قانونيًا
LEGAL_DEADLINES = {
    "تقادم دعوى مدنية": 15,
    "تقادم دعوى تجارية": 10,
    "الطعن في صفقة عمومية (بعد التبليغ)": 60,
    "الطعن في قرار إداري": 30,
    "إنهاء عقد عمل (إشعار مسبق)": 30,
}

@app.route('/calculate_deadlines', methods=['POST'])
def calculate_deadlines():
    """واجهة API لحساب المواعيد القانونية: تستقبل الإجراء وتاريخ البدء، وتعيد تاريخ الانتهاء والأيام المتبقية."""
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

# ---------------------- مسارات أساسية أخرى ----------------------
@app.route('/')
def serve_index():
    """تقديم ملف index.html من المسار الحالي (الصفحة الرئيسية للتطبيق)."""
    return send_from_directory('.', 'index.html')

@app.route('/ask', methods=['POST'])
def ask():
    """واجهة API للأسئلة القانونية: تستقبل استعلام المستخدم وتعيد الإجابة عبر ask_lawyer."""
    data = request.get_json()
    query = data.get('query', '')
    if not query:
        return jsonify({"error": "الرجاء إدخال سؤال"}), 400
    return jsonify(ask_lawyer(query))

@app.route('/upload', methods=['POST'])
def upload():
    """واجهة API لرفع ملف (PDF أو صورة) وإضافته إلى مكتبة المتجهات."""
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
    """واجهة API لإحصائيات المكتبة: تعيد عدد المقاطع (chunks) المخزنة في قاعدة المتجهات."""
    return jsonify({"chunks_count": collection.count()})

# ---------------------- تشغيل الخادم ----------------------
if __name__ == '__main__':
    # التأكد من وجود مجلد البيانات الأساسي (يمكن إضافة ملفات نموذجية فيه)
    os.makedirs("data/contrats_exemples", exist_ok=True)
    # بناء المكتبة تلقائيًا من المجلدات المحددة إذا كانت فارغة
    initial_scan_and_build()
    # تشغيل خادم Flask على المنفذ 5000، مع جعله متاحًا على جميع واجهات الشبكة
    app.run(host='0.0.0.0', port=5000, debug=False)