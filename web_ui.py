import streamlit as st
import os
import sys
import chromadb
import pdfplumber
import json
import re
import tempfile
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import time

# ========== إعدادات الصفحة ==========
st.set_page_config(
    page_title="المستشار القانوني الجزائري",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تطبيق الأنماط العربية والتحسينات البصرية
st.markdown("""
<style>
    /* الاتجاه العام */
    .stApp {
        direction: rtl;
    }

    /* تنسيق الشريط الجانبي */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    /* تنسيق مربع الدردشة */
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        max-width: 80%;
        align-self: flex-end;
        margin-left: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .bot-message {
        background: white;
        color: #1e293b;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        max-width: 80%;
        align-self: flex-start;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .source-box {
        background: #fff8e7;
        border-right: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #5d4037;
    }
    .source-title {
        color: #ff9800;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .stTextInput > div > input {
        direction: rtl;
        text-align: right;
        border-radius: 25px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
    }
    /* تنسيق رافع الملفات */
    .stFileUploader > div {
        direction: rtl;
    }
    .uploadedFile {
        background: #f0f9ff;
        border-radius: 10px;
        padding: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ========== تهيئة النظام مع التخزين المؤقت ==========
@st.cache_resource
def init_system():
    """تحميل النماذج والاتصال بقاعدة البيانات (مرة واحدة فقط)"""
    # محاولة قراءة مفتاح API من متغير البيئة
    api_key = os.getenv("GEMINI_API_KEY")

    # تهيئة عميل Gemini (بدون مفتاح إذا لم يوجد)
    client = genai.Client(api_key=api_key) if api_key else None

    # تحميل نموذج التضمين
    model_embedding = SentenceTransformer('intfloat/multilingual-e5-small')

    # الاتصال بقاعدة البيانات
    client_db = chromadb.PersistentClient(path="legal_db")

    # التحقق من وجود المجموعة
    try:
        collection = client_db.get_collection(name="algerian_law")
    except ValueError:
        st.warning("⚠️ المجموعة 'algerian_law' غير موجودة. سيتم إنشاؤها...")
        collection = client_db.create_collection(name="algerian_law")

    return client, model_embedding, collection


# ========== دوال معالجة النصوص (مأخوذة من main_ai.py) ==========
def force_fix_arabic(text):
    if not text:
        return ""
    lines = text.split('\n')
    fixed_lines = []
    for line in lines:
        line = line.strip()
        if re.search(r'[\u0600-\u06FF]', line) and "ةداملا" in line:
            fixed_line = line[::-1]
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    return " ".join(fixed_lines)


# ========== دوال إدارة السجل ==========
def load_history():
    history_file = "processed_history.json"
    if os.path.exists(history_file):
        with open(history_file, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_history(history):
    history_file = "processed_history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(list(history), f, ensure_ascii=False, indent=2)


# ========== دالة إضافة ملف PDF إلى المكتبة ==========
def add_pdf_to_library(pdf_path, model_embedding, collection):
    rel_path = os.path.basename(pdf_path)  # اسم الملف فقط للعرض
    history = load_history()

    if rel_path in history:
        return f"⚠️ الملف '{rel_path}' تمت إضافته مسبقاً."

    try:
        # استخراج النص من PDF
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=2)
                except:
                    text = page.extract_text()
                if text:
                    full_text += text + "\n"

        if not full_text.strip():
            return "❌ الملف لا يحتوي على نصوص قابلة للاستخراج."

        # تنظيف وتقطيع النص
        clean_text = force_fix_arabic(full_text)
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
        if buffer:
            chunk_text = " ".join(buffer)
            chunks.append(chunk_text)

        if not chunks:
            return "❌ لم يتم استخراج أي نصوص."

        # حساب المتجهات والإضافة
        embeddings = []
        ids = []
        metadatas = []
        for idx, chunk in enumerate(chunks):
            vector = model_embedding.encode("passage: " + chunk).tolist()
            embeddings.append(vector)
            ids.append(f"{rel_path}_part{idx + 1}")
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


# ========== دالة البحث والإجابة ==========
def ask_lawyer(query, model_embedding, collection, client):
    query_vector = model_embedding.encode("query: " + query).tolist()

    try:
        results = collection.query(query_embeddings=[query_vector], n_results=5)
    except Exception as e:
        return f"⚠️ خطأ في البحث: {e}", None

    if not results['documents'][0]:
        return "عذراً، لم أتمكن من العثور على معلومات.", None

    context = "\n\n".join(results['documents'][0])

    system_prompt = """أنت مستشار قانوني جزائري خبير.
    مهمتك هي الإجابة على أسئلة المستخدمين بناءً على النصوص القانونية المقدمة فقط.

    تعليمات مهمة:
    - قدم إجابة شاملة وكاملة دون اختصار.
    - في الأسئلة العادية، اذكر المصدر باختصار (مثل: "المادة 5 من القانون التجاري").
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
        return response.text, results
    except Exception as e:
        return f"⚠️ خطأ في الاتصال: {str(e)}", None


# ========== تهيئة النظام (مرة واحدة) ==========
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.messages = []  # لتخزين المحادثة
    st.session_state.last_results = None
    st.session_state.api_key_set = False

# تحميل النظام
client, model_embedding, collection = init_system()

# ========== الشريط الجانبي ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/justice-scales.png", width=80)
    st.markdown("<h1 style='color: white; text-align: center;'>⚖️ المستشار القانوني</h1>", unsafe_allow_html=True)
    st.markdown("---")

    # إدخال مفتاح API
    with st.expander("🔑 إعدادات API", expanded=not st.session_state.api_key_set):
        api_key_input = st.text_input("أدخل مفتاح Gemini API", type="password", key="api_key")
        if api_key_input:
            os.environ["GEMINI_API_KEY"] = api_key_input
            client = genai.Client(api_key=api_key_input)
            st.session_state.api_key_set = True
            st.success("✅ تم تعيين المفتاح")
            st.rerun()

    # رفع ملفات PDF
    st.markdown("---")
    st.markdown("<h3 style='color: white;'>📂 إضافة ملفات</h3>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("اختر ملفات PDF", type=['pdf'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            with st.spinner(f"جاري معالجة {uploaded_file.name}..."):
                result = add_pdf_to_library(tmp_path, model_embedding, collection)
                if "✅" in result:
                    st.success(result)
                else:
                    st.warning(result)
            os.unlink(tmp_path)  # حذف الملف المؤقت

    # إحصائيات سريعة
    st.markdown("---")
    try:
        count = collection.count()
        st.markdown(f"<p style='color: white;'>📊 عدد الأجزاء في المكتبة: {count}</p>", unsafe_allow_html=True)
    except:
        pass

# ========== منطقة المحادثة الرئيسية ==========
st.markdown("<h1 style='text-align: center; color: #1e3c72;'>⚖️ نظام الاستشارات القانونية الجزائرية</h1>",
            unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>اسأل أي سؤال قانوني وسأجيبك بالاعتماد على النصوص الرسمية</p>",
            unsafe_allow_html=True)

# عرض تاريخ المحادثة
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">{msg["content"]}</div>', unsafe_allow_html=True)
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 عرض المصادر"):
                    for i, (doc, src) in enumerate(zip(msg["sources"]["documents"][0], msg["sources"]["metadatas"][0])):
                        st.markdown(
                            f'<div class="source-box"><div class="source-title">المصدر {i + 1}: {src["source"]}</div>{doc[:500]}...</div>',
                            unsafe_allow_html=True)

# مربع إدخال السؤال
col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("اكتب سؤالك هنا:", key="input", placeholder="مثال: ما هو السجل التجاري؟")
with col2:
    submit = st.button("إرسال", use_container_width=True)

if submit and user_input:
    # إضافة سؤال المستخدم إلى المحادثة
    st.session_state.messages.append({"role": "user", "content": user_input})

    # التحقق من وجود مفتاح API
    if not st.session_state.api_key_set and not os.getenv("GEMINI_API_KEY"):
        response = "❌ يرجى إدخال مفتاح Gemini API في الشريط الجانبي أولاً."
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.spinner("🤖 جاري البحث والتحليل..."):
            answer, results = ask_lawyer(user_input, model_embedding, collection, client)
            if results:
                st.session_state.last_results = results
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
            else:
                st.session_state.messages.append({"role": "assistant", "content": answer})

    st.rerun()

# أزرار إضافية أسفل الصفحة
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🧹 مسح المحادثة"):
        st.session_state.messages = []
        st.rerun()
with col2:
    if st.button("📋 عرض آخر المصادر") and st.session_state.last_results:
        sources_text = ""
        for i, (doc, src) in enumerate(
                zip(st.session_state.last_results['documents'][0], st.session_state.last_results['metadatas'][0])):
            sources_text += f"**المصدر {i + 1}:** {src['source']}\n\n{doc[:300]}...\n\n---\n\n"
        st.info(sources_text)
with col3:
    if st.button("ℹ️ حول النظام"):
        st.markdown("""
        **نظام الاستشارات القانونية الجزائري**  
        - يعتمد على تقنية RAG لاسترجاع المعلومات القانونية.  
        - قاعدة البيانات تحتوي على نصوص قانونية جزائرية رسمية.  
        - يستخدم نموذج Gemini للإجابة على الأسئلة.  
        - يمكنك إضافة ملفات PDF جديدة عبر الشريط الجانبي.
        """)