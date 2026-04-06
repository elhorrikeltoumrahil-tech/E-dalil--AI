import os
import re
import fitz  # PyMuPDF
from paddleocr import PaddleOCR

class SmartDocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.full_text = ""
        # تهيئة PaddleOCR للغة العربية لاستخدامها عند الحاجة
        self.ocr = PaddleOCR(lang='ar', use_angle_cls=True, use_gpu=False)
        # تعبير نمطي للبحث عن عناوين المواد (مثال: "المادة 1", "الفصل الثاني")
        self.heading_pattern = re.compile(r'(المادة|الفصل|القسم|الفرع)\s+(\d+|الأول|الثاني|ثالث)')

    def extract_text(self):
        """استخراج النص من ملف PDF"""
        doc = fitz.open(self.pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                # إذا كانت الصفحة نصية، أضف النص مباشرة
                self.full_text += page_text + "\n"
            else:
                # إذا كانت الصفحة ممسوحة ضوئياً، استخدم PaddleOCR
                pix = page.get_pixmap()
                img_path = f"temp_page_{page_num}.png"
                pix.save(img_path)
                result = self.ocr.ocr(img_path, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        self.full_text += line[1][0] + " "
                os.remove(img_path)
        return self.full_text

    def smart_chunk(self, text):
        """تقطيع النص بناءً على العناوين (المواد، الفصول، إلخ)"""
        chunks = []
        current_chunk = ""
        lines = text.split('\n')
        for line in lines:
            # إذا وجدنا سطراً يبدو كعنوان، نبدأ قطعة جديدة
            if self.heading_pattern.search(line):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = line + "\n"
            else:
                current_chunk += line + "\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def process(self):
        """تنفيذ العملية الكاملة"""
        print(f"⏳ جاري معالجة الملف: {self.pdf_path}")
        full_text = self.extract_text()
        chunks = self.smart_chunk(full_text)
        print(f"✅ تم استخراج {len(chunks)} قطعة نصية.")
        return chunks