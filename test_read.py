import pdfplumber

# تأكد أن اسم الملف صحيح
pdf_file = "law.pdf"

try:
    with pdfplumber.open(pdf_file) as pdf:
        # قراءة الصفحة الأولى
        if len(pdf.pages) > 0:
            first_page = pdf.pages[0]
            text = first_page.extract_text()

            print("--- بداية النص ---")
            print(text)
            print("--- نهاية النص ---")

            if not text or text.strip() == "":
                print("\n⚠️ النتيجة: الصفحة فارغة! (الملف عبارة عن صور Scanned)")
            else:
                print("\n✅ النتيجة: تم العثور على نص. تأكد هل كلمة 'المادة' مكتوبة بوضوح؟")
        else:
            print("الملف لا يحتوي على صفحات!")

except FileNotFoundError:
    print(f"خطأ: الملف {pdf_file} غير موجود. تأكد من الاسم والمكان.")
except Exception as e:
    print(f"حدث خطأ آخر: {e}")