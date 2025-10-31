# ⚡ دليل البدء السريع (Quick Start Guide)

دليل سريع للبدء مع StuTally في أقل من 5 دقائق!

---

## 📋 المتطلبات الأساسية

تأكد من توفر:
- Python 3.8+ مثبت
- Git (للاستنساخ)
- 5GB مساحة فارغة

---

## 🚀 البدء السريع

### الخطوة 1: الاستنساخ
```bash
git clone https://github.com/aseelalmutari/StuTally-Project.git
cd StuTally
```

### الخطوة 2: البيئة الافتراضية
```bash
python -m venv venv
source venv/bin/activate  # على Windows: venv\Scripts\activate
```

### الخطوة 3: التثبيت
```bash
pip install -r requirements.txt
```

### الخطوة 4: الإعداد
```bash
# نسخ ملف البيئة
cp .env.example .env

# إنشاء مستخدم admin
python create_user.py
```

### الخطوة 5: التشغيل
```bash
python app.py
```

✅ **جاهز!** افتح http://localhost:5000

---

## 🎯 الاستخدام السريع

### رفع فيديو
1. افتح http://localhost:5000
2. اختر ملف فيديو
3. اختر النموذج (yolov8s.pt)
4. فعّل "Enable Counting Line" (اختياري)
5. اضغط "Upload"
6. شاهد المعالجة الحية!

### الوصول للتحليلات
1. اذهب إلى http://localhost:5000/login
2. استخدم: `admin` / `admin`
3. ستُحوّل تلقائياً للتحليلات

---

## 🔧 مشاكل شائعة

### المشكلة: خطأ في التثبيت
```bash
# الحل
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### المشكلة: النموذج لا يعمل
```bash
# سيتم التحميل تلقائياً، أو:
# ضع ملفات .pt في مجلد models/
```

### المشكلة: المنفذ 5000 مُستخدم
```bash
# على macOS (AirPlay)
# System Preferences > Sharing > أوقف AirPlay Receiver

# أو غيّر المنفذ في .env
PORT=5001
```

---

## 📚 الخطوات التالية

- 📖 [التوثيق الكامل](../README.md)
- 🤝 [دليل المساهمة](../CONTRIBUTING.md)
- 🐛 [الإبلاغ عن مشكلة](https://github.com/aseelalmutari/StuTally-Project/issues)

---

**🎉 مبروك! أنت الآن جاهز لاستخدام StuTally!**

