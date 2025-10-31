# 📥 دليل تحميل النماذج - StuTally Models Download Guide

> **ملاحظة مهمة**: نماذج YOLO غير مضمنة في مستودع GitHub بسبب حجمها الكبير

---

## 🎯 نظرة سريعة

نماذج YOLO المستخدمة في StuTally تتراوح أحجامها من **6MB إلى 400MB+**، لذلك لم يتم تضمينها في المستودع.

### ✅ ما تحتاج لفعله

1. تحميل النماذج من الرابط الخارجي (أنظر أدناه)
2. وضع الملفات `.pt` في مجلد `models/`
3. تشغيل التطبيق

---

## 📦 روابط التحميل

### 🔗 الرابط الرئيسي

<div align="center">

## **[📥 تحميل جميع النماذج من هنا](https://drive.google.com/drive/folders/1FbXiXZtd6Zf8A_8d77Tz5kswGu5m-whX?usp=share_link)**

**Google Drive - StuTally Models Repository**

---

### أو استخدم الروابط المباشرة:

</div>

| النموذج | الحجم | الوصف | رابط مباشر |
|---------|------|-------|------------|
| **yolov8s.pt** | 11MB | النموذج الافتراضي ⭐ | [تحميل](رابط) |
| yolov8m.pt | 26MB | نموذج متوسط | [تحميل](رابط) |
| yolov8l.pt | 44MB | نموذج كبير | [تحميل](رابط) |
| yolov11x.pt | 136MB | نموذج v11 | [تحميل](رابط) |
| **best.pt** | ؟؟ MB | نموذج مخصص ⭐ | [تحميل](رابط) |

---

## 🚀 خطوات التثبيت السريعة

### 1️⃣ تحميل النماذج

**الخيار أ: تحميل الكل (موصى به)**
```bash
# حمّل ملف مضغوط يحتوي على جميع النماذج
# من الرابط الرئيسي أعلاه
```

**الخيار ب: تحميل نموذج محدد**
```bash
# حمّل فقط النموذج الذي تحتاجه
# من الروابط المباشرة أعلاه
```

### 2️⃣ فك الضغط (إن كان مضغوطاً)

```bash
# إذا كان الملف مضغوطاً (.zip, .tar.gz)
unzip models.zip
# أو
tar -xzf models.tar.gz
```

### 3️⃣ نقل إلى مجلد models

```bash
cd /Users/a1443/Desktop/StuTally

# نسخ النماذج
cp ~/Downloads/yolov8s.pt models/
cp ~/Downloads/best.pt models/

# أو نسخ الكل
cp ~/Downloads/*.pt models/

# التحقق
ls -lh models/*.pt
```

### 4️⃣ التحقق من النجاح

```bash
# يجب أن ترى:
# models/yolov8s.pt
# models/best.pt (اختياري)
# models/README.md
# models/.gitkeep

ls -lh models/
```

---

## ⚡ الأوامر المختصرة

### تحميل وتثبيت بأمر واحد (إذا كان لديك رابط مباشر)

```bash
cd /Users/a1443/Desktop/StuTally/models

# باستخدام wget
wget -O yolov8s.pt "رابط_التحميل_المباشر"

# أو باستخدام curl  
curl -L -o yolov8s.pt "رابط_التحميل_المباشر"
```

### تحميل من Google Drive باستخدام gdown

```bash
# تثبيت gdown
pip install gdown

# تحميل ملف واحد
cd models/
gdown "FILE_ID_FROM_GOOGLE_DRIVE"

# أو تحميل مجلد كامل
gdown --folder "FOLDER_ID_FROM_GOOGLE_DRIVE"
```

---

## 📋 النماذج المطلوبة

### الحد الأدنى للتشغيل

✅ **yolov8s.pt** - مطلوب للتشغيل الأساسي

### موصى به

✅ **yolov8s.pt** - للاستخدام العام  
✅ **best.pt** - للنموذج المخصص (تصنيف المراحل الدراسية)

### اختياري (للدقة الأعلى)

⭕ yolov8m.pt  
⭕ yolov8l.pt  
⭕ yolov11x.pt

---

## 🎓 معلومات إضافية

### لماذا لم يتم تضمين النماذج؟

1. **الحجم الكبير**: النماذج كبيرة جداً لـ Git
2. **أفضل ممارسات**: Git مخصص للكود، ليس للملفات الكبيرة
3. **سهولة التحديث**: يمكن تحديث النماذج بشكل منفصل
4. **المرونة**: المستخدمون يختارون النماذج التي يحتاجونها فقط

### التحميل التلقائي

النماذج القياسية من Ultralytics (yolov8n, yolov8s, إلخ) **سيتم تحميلها تلقائياً** عند أول استخدام إذا:
- لم تكن موجودة في مجلد `models/`
- لديك اتصال بالإنترنت

⚠️ **لكن**: النموذج المخصص `best.pt` يجب تحميله يدوياً.

---

## 🔧 استكشاف الأخطاء

### المشكلة: "Model not found"

```bash
# تأكد من وجود النموذج
ls -l models/yolov8s.pt

# إذا لم يكن موجوداً، حمّله من الرابط أعلاه
```

### المشكلة: "Permission denied"

```bash
# تعديل الصلاحيات
chmod 644 models/*.pt
```

### المشكلة: "Corrupted file"

```bash
# حذف وإعادة التحميل
rm models/yolov8s.pt
# ثم حمّل من جديد
```

### المشكلة: بطء التحميل

**حلول**:
1. استخدم download manager
2. استخدم VPN إذا كانت السرعة بطيئة
3. حمّل النموذج الأصغر أولاً (yolov8n أو yolov8s)
4. استخدم `aria2c` لتحميل أسرع:
   ```bash
   aria2c -x 16 -s 16 "رابط_التحميل"
   ```

---

## 📚 مراجع إضافية

- 📖 [README.md الرئيسي](README.md) - التوثيق الكامل
- 📁 [models/README.md](models/README.md) - دليل تفصيلي للنماذج
- 🚀 [docs/QUICK_START.md](docs/QUICK_START.md) - دليل البدء السريع
- 🔧 [CONTRIBUTING.md](CONTRIBUTING.md) - دليل المساهمة

---

## 📞 المساعدة

إذا واجهت مشكلة:

1. 💬 افتح [GitHub Discussion](https://github.com/aseelalmutari/StuTally-Project/discussions)
2. 🐛 أبلغ عن [Issue](https://github.com/aseelalmutari/StuTally-Project/issues)
3. 📧 راسلنا: support@stutally.project

---

## ✅ Checklist

بعد تحميل النماذج، تأكد من:

- [ ] النماذج موجودة في `models/*.pt`
- [ ] الصلاحيات صحيحة (`644`)
- [ ] الملفات غير تالفة (يمكن فتحها)
- [ ] المساحة كافية (~300MB للكل)

---

<div align="center">

## 🎉 جاهز للانطلاق!

بعد تحميل النماذج، يمكنك البدء:

```bash
python app.py
```

ثم افتح: http://localhost:5000

---

**أسئلة؟ اتصل بنا!**

📧 support@stutally.project

</div>

---

**آخر تحديث**: 31 أكتوبر 2025

