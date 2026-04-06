import os
import sys
import chromadb
import pdfplumber
import json
import re
import arabic_reshaper
from bidi.algorithm import get_display
import tkinter as tk
from tkinter import filedialog
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types

# ========== إعدادات الملفات والسجلات ==========
HISTORY_FILE = "processed_history.json"
PDF_FOLDER = "pdfs_to_add"  # مجلد اختياري لوضع ملفات PDF للإضافة السريعة

# متغير عمومي لحفظ آخر نتائج البحث
last_results = None

# ========== دوال إدارة السجل ==========
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(list(history), f, ensure_ascii=False, indent=2)

# ========== دوال معالجة النصوص العربية ==========
def force_fix_arabic(text):
    """
    دالة احترافية لمعالجة النص العربي المستخرج من PDF:
    1. تربط الحروف المتقطعة
    2. تصحح اتجاه النص (يمين-يسار) مع الحفاظ على ترتيب الكلمات والأرقام
    """
    if not text:
        return ""

    try:
        # 1. إعادة ربط الحروف العربية لتكون متصلة
        reshaped_text = arabic_reshaper.reshape(text)

        # 2. تعديل الاتجاه الصحيح للغة العربية دون قلب الأرقام أو الكلمات الأجنبية
        bidi_text = get_display(reshaped_text)

        # 3. تنظيف الفراغات والأسطر ليكون النص عبارة عن فقرات متصلة يفهمها الذكاء الاصطناعي
        lines = bidi_text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]

        return " ".join(cleaned_lines)

    except Exception as e:
        print(f"⚠️ خطأ أثناء معالجة النص العربي: {e}")
        return text

# ========== دالة اختيار ملف PDF ==========
def select_pdf_file():
    """فتح نافذة اختيار ملف PDF وإرجاع المسار المحدد"""
    root = tk.Tk()
    root.withdraw()  # إخفاء النافذة الرئيسية
    root.attributes('-topmost', True)  # جعلها في المقدمة
    file_path = filedialog.askopenfilename(
        title="اختر ملف PDF",
        filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path if file_path else None

# ========== دالة إضافة ملف PDF جديد ==========
def add_pdf_to_library(pdf_path):
    """
    تقوم باستخراج النص من ملف PDF، تقسيمه إلى أجزاء، حساب المتجهات،
    وإضافتها إلى قاعدة بيانات ChromaDB، مع تحديث سجل الملفات المضافة.
    """
    global model_embedding, collection

    # التحقق من وجود الملف
    if not os.path.exists(pdf_path):
        return f"❌ الملف '{pdf_path}' غير موجود."

    rel_path = os.path.relpath(pdf_path)  # المسار النسبي للتخزين في السجل
    history = load_history()

    # التحقق من أن الملف لم يضف من قبل
    if rel_path in history:
        return f"⚠️ الملف '{rel_path}' تمت إضافته مسبقاً."

    print(f"\n📄 جاري معالجة الملف: {rel_path}...")

    try:
        # استخراج النص من جميع صفحات PDF
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                except:
                    text = page.extract_text()
                if text:
                    full_text += text + "\n"
                else:
                    print(f"⚠️ الصفحة {page_num+1} لا تحتوي على نص قابل للاستخراج (قد تكون صورة).")

        if not full_text.strip():
            return "❌ الملف لا يحتوي على نصوص قابلة للاستخراج (قد يكون ممسوحاً ضوئياً)."

        # تنظيف النص وإصلاح العربية المقلوبة
        clean_text = force_fix_arabic(full_text)

        # تقسيم النص إلى أجزاء (chunks) بحجم ~500 كلمة
        words = clean_text.split()
        chunks = []
        buffer = []
        current_len = 0
        part_counter = 1

        for word in words:
            buffer.append(word)
            current_len += len(word) + 1
            if current_len >= 500:
                chunk_text = " ".join(buffer)
                chunks.append(chunk_text)
                buffer = []
                current_len = 0
                part_counter += 1

        if buffer:  # إضافة الجزء الأخير
            chunk_text = " ".join(buffer)
            chunks.append(chunk_text)

        if not chunks:
            return "❌ لم يتم استخراج أي نصوص من الملف."

        # حساب المتجهات وإضافتها إلى ChromaDB
        print(f"⏳ جاري تحويل {len(chunks)} جزء إلى متجهات...")
        embeddings = []
        ids = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            # حساب المتجه باستخدام بادئة passage: كما هو مطلوب لنموذج e5
            vector = model_embedding.encode("passage: " + chunk).tolist()
            embeddings.append(vector)
            ids.append(f"{rel_path}_part{idx+1}")
            metadatas.append({"source": rel_path})

        # إضافة إلى المجموعة
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        # تحديث السجل التاريخي
        history.add(rel_path)
        save_history(history)

        return f"✅ تمت إضافة {len(chunks)} جزء من الملف '{rel_path}' بنجاح."

    except Exception as e:
        return f"❌ حدث خطأ أثناء معالجة الملف: {e}"

# ========== تهيئة النظام ==========
# قراءة مفتاح Gemini API من متغير البيئة
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("❌ خطأ: لم يتم تعيين مفتاح Gemini API. يرجى تعيين متغير البيئة GEMINI_API_KEY.")
    sys.exit(1)

# تهيئة عميل Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

print("⏳ جاري تحميل النظام...")
model_embedding = SentenceTransformer('intfloat/multilingual-e5-small')
client_db = chromadb.PersistentClient(path="legal_db")

# التحقق من وجود المجموعة أو إنشاؤها
try:
    collection = client_db.get_collection(name="algerian_law")
except ValueError:
    print("⚠️ المجموعة 'algerian_law' غير موجودة. سيتم إنشاؤها الآن...")
    collection = client_db.create_collection(name="algerian_law")

# ========== دالة البحث والإجابة ==========
def ask_lawyer(query):
    global last_results
    query_vector = model_embedding.encode("query: " + query).tolist()

    try:
        results = collection.query(query_embeddings=[query_vector], n_results=5)
        last_results = results  # حفظ النتائج للاستخدام لاحقاً
    except Exception as e:
        return f"⚠️ خطأ أثناء البحث في قاعدة البيانات: {e}"

    if not results['documents'][0]:
        return "عذراً، لم أتمكن من العثور على معلومات متعلقة بسؤالك."

    # بناء السياق بدون ذكر المصادر (فقط النصوص)
    context = "\n\n".join(results['documents'][0])

    system_prompt = """أنت مستشار قانوني جزائري خبير.
    مهمتك هي الإجابة على أسئلة المستخدمين بناءً على النصوص القانونية المقدمة فقط.

    تعليمات مهمة:
    - قدم إجابة شاملة وكاملة دون اختصار.
    - إذا طلب المستخدم مصدر المعلومة (مثلاً: "أين المصدر؟" أو "من أين هذه المعلومة؟")، قم بعرض الفقرة القانونية كاملة كما وردت في النصوص المقدمة (حوالي 5 إلى 10 أسطر) مع ذكر اسم الملف المصدر.
    - في الأسئلة العادية، اذكر المصدر باختصار (مثل: "المادة 5 من القانون التجاري") ولكن ركز على شرح الإجابة.
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
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "API key" in error_msg:
            return "❌ مفتاح API غير صالح. تحقق من صحة المفتاح."
        elif "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
            return "⚠️ تجاوزت حد الاستخدام المسموح به. حاول مرة أخرى لاحقاً."
        else:
            return f"⚠️ حدث خطأ غير متوقع أثناء التواصل مع Gemini: {error_msg}"

# ========== دالة عرض المصادر ==========
def show_source():
    """عرض المصادر (النصوص الكاملة) لآخر بحث"""
    if last_results is None or not last_results['documents'][0]:
        return "لا توجد معلومات مصدر متاحة حالياً."
    output = "📚 المصادر المستخدمة في الإجابة الأخيرة:\n\n"
    for i, doc in enumerate(last_results['documents'][0]):
        src = last_results['metadatas'][0][i]['source']
        # عرض أول 10 أسطر أو 500 حرف تقريباً
        preview = "\n".join(doc.split("\n")[:10])
        if len(preview) < len(doc):
            preview += "\n..."
        output += f"--- المصدر {i+1}: {src} ---\n{preview}\n\n"
    return output

# ========== الواجهة الرئيسية ==========
if __name__ == "__main__":
    print("\n⚖️  نظام الاستشارات القانونية الجزائرية (Gemini) ⚖️")
    print("الأوامر المتاحة:")
    print("  - اكتب سؤالك القانوني للحصول على إجابة.")
    print("  - اكتب 'مصدر' لعرض المصادر من آخر إجابة.")
    print("  - اكتب 'اضف ملف' فقط لفتح نافذة اختيار ملف PDF.")
    print("  - أو اكتب 'اضف ملف: مسار_الملف' لإضافة ملف محدد (مثال: اضف ملف: قانون.pdf).")
    print("  - اكتب 'اضف مجلد' لمعالجة جميع ملفات PDF في مجلد 'pdfs_to_add' (إذا كان موجوداً).")
    print("  - اكتب 'exit' للخروج.\n")

    while True:
        q = input("أنت: ").strip()
        if q.lower() == "exit":
            break
        if not q:
            continue

        # أمر عرض المصادر
        if q.strip().lower() in ["مصدر", "المصدر", "source"]:
            print(show_source())
            continue

        # أمر إضافة ملف PDF
        if q.startswith("اضف ملف") or q.startswith("إضافة ملف"):
            if q.strip() in ["اضف ملف", "إضافة ملف"]:
                # فتح نافذة اختيار الملف
                file_path = select_pdf_file()
                if file_path:
                    result = add_pdf_to_library(file_path)
                    print(result)
                else:
                    print("❌ لم يتم اختيار أي ملف.")
            else:
                # استخدام المسار المكتوب بعد النقطتين
                parts = q.split(":", 1)
                if len(parts) < 2:
                    print("❌ الصيغة الصحيحة: اضف ملف: مسار_الملف  أو  اضف ملف فقط لاختيار ملف")
                    continue
                file_path = parts[1].strip()
                result = add_pdf_to_library(file_path)
                print(result)
            continue

        # أمر إضافة مجلد كامل
        if q.strip().lower() in ["اضف مجلد", "إضافة مجلد"]:
            if not os.path.exists(PDF_FOLDER):
                print(f"❌ المجلد '{PDF_FOLDER}' غير موجود. قم بإنشائه وضع ملفات PDF داخله.")
                continue
            pdf_files = [os.path.join(PDF_FOLDER, f) for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
            if not pdf_files:
                print(f"⚠️ لا توجد ملفات PDF في مجلد '{PDF_FOLDER}'.")
                continue
            print(f"📦 تم العثور على {len(pdf_files)} ملف PDF في المجلد.")
            for pdf_file in pdf_files:
                result = add_pdf_to_library(pdf_file)
                print(result)
            continue

        # وإلا فهو سؤال عادي
        print("\n🤖 جاري البحث والتحليل...")
        answer = ask_lawyer(q)
        print(f"\n🤖 الإجابة:\n{answer}\n")