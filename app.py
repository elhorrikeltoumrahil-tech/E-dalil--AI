import chromadb
from sentence_transformers import SentenceTransformer

# 1. الاتصال بقاعدة البيانات
client = chromadb.PersistentClient(path="legal_db")
collection = client.get_collection(name="algerian_law")

# 2. تحميل نموذج الفهم (نفس النموذج المستخدم في التخزين)
print("⏳ جاري تحميل نموذج الذكاء الاصطناعي...")
model = SentenceTransformer('intfloat/multilingual-e5-small')


def search(query):
    print(f"\n🔍 جاري البحث عن: '{query}'...")

    # تحويل السؤال إلى أرقام
    query_vector = model.encode("query: " + query).tolist()

    # البحث عن أقرب 3 فقرات
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )

    documents = results['documents'][0]
    ids = results['ids'][0]

    if not documents:
        print("❌ لم يتم العثور على نتائج.")
        return

    print(f"\n✅ وجدنا {len(documents)} نتائج ذات صلة:\n")
    for i in range(len(documents)):
        print(f"--- النتيجة {i + 1} (المصدر: {ids[i]}) ---")
        print(documents[i])
        print("-" * 40)


# --- واجهة التجربة ---
if __name__ == "__main__":
    print("\n⚖️  مرحباً بك في نظام البحث القانوني الجزائري ⚖️")
    while True:
        q = input("\nأدخل سؤالك القانوني (أو exit للخروج): ")
        if q.lower() == "exit":
            break
        search(q)