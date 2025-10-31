# 📦 نماذج YOLO - دليل التحميل

هذا المجلد يحتوي على نماذج YOLO المستخدمة في StuTally للكشف عن الأشخاص والتصنيف.

---

## 🔗 تحميل النماذج

### ⚠️ ملاحظة مهمة

بسبب الحجم الكبير لنماذج YOLO (**من 6MB إلى 400MB+**)، لا يتم تضمينها في مستودع GitHub.

### 📥 روابط التحميل

#### 🌐 الرابط الرئيسي (موصى به)

**[📥 تحميل جميع النماذج - Google Drive](https://drive.google.com/drive/folders/1FbXiXZtd6Zf8A_8d77Tz5kswGu5m-whX?usp=share_link)**

أو

**[📥 تحميل النماذج - Dropbox](رابط_dropbox_هنا)**

أو

**[📥 تحميل النماذج - OneDrive](رابط_onedrive_هنا)**

#### 🔗 روابط مباشرة للنماذج الفردية

إذا كنت تريد تحميل نموذج محدد فقط:

| النموذج | الحجم | الوصف | رابط التحميل |
|---------|------|-------|---------------|
| `yolov8s.pt` | ~11MB | النموذج الافتراضي (موصى به) | [تحميل](رابط_مباشر) |
| `yolov8m.pt` | ~26MB | نموذج متوسط (دقة أعلى) | [تحميل](رابط_مباشر) |
| `yolov8l.pt` | ~44MB | نموذج كبير (دقة عالية) | [تحميل](رابط_مباشر) |
| `yolov11x.pt` | ~136MB | نموذج v11 كبير جداً | [تحميل](رابط_مباشر) |
| `best.pt` | ~تحديد الحجم~ | **النموذج المخصص** (تصنيف المراحل) | [تحميل](رابط_مباشر) |

---

## 📂 التثبيت

### الطريقة 1: التحميل اليدوي

1. **حمّل النماذج** من الروابط أعلاه
2. **فك الضغط** (إن كانت مضغوطة)
3. **انقل الملفات** إلى مجلد `models/`:

```bash
# من مجلد التحميل
cp ~/Downloads/yolov8s.pt /Users/youuser/Desktop/StuTally/models/
cp ~/Downloads/best.pt /Users/youuser/Desktop/StuTally/models/

# أو باستخدام مسار نسبي
cd /Users/youuser/Desktop/StuTally
cp ~/Downloads/*.pt models/
```

### الطريقة 2: استخدام wget/curl (إذا كان لديك رابط مباشر)

```bash
cd models/

# باستخدام wget
wget -O yolov8s.pt "رابط_تحميل_مباشر"

# أو باستخدام curl
curl -L -o yolov8s.pt "رابط_تحميل_مباشر"
```

### الطريقة 3: استخدام gdown (للتحميل من Google Drive)

```bash
# تثبيت gdown
pip install gdown

# تحميل من Google Drive
cd models/
gdown "google_drive_file_id"

# أو تحميل مجلد كامل
gdown --folder "google_drive_folder_id"
```

---

## ✅ التحقق من التثبيت

بعد التحميل، تأكد من وجود النماذج:

```bash
cd models/
ls -lh

# يجب أن ترى:
# -rw-r--r--  1 user  staff   11M  yolov8s.pt
# -rw-r--r--  1 user  staff   26M  yolov8m.pt
# -rw-r--r--  1 user  staff   44M  yolov8l.pt
# -rw-r--r--  1 user  staff  136M  yolov11x.pt
# -rw-r--r--  1 user  staff   XXM  best.pt
```

أو باستخدام Python:

```python
import os
from pathlib import Path

models_dir = Path(__file__).parent
models = list(models_dir.glob('*.pt'))

print("النماذج المثبتة:")
for model in models:
    size_mb = model.stat().st_size / (1024 * 1024)
    print(f"  ✓ {model.name} ({size_mb:.1f} MB)")
```

---

## 📋 النماذج المطلوبة

### الحد الأدنى (Essential)

- ✅ **yolov8s.pt** - النموذج الافتراضي (مطلوب)

### موصى به (Recommended)

- ✅ **yolov8s.pt** - للاستخدام العام
- ✅ **best.pt** - للتصنيف المخصص (تصنيف المراحل الدراسية)

### اختياري (Optional)

- ⭕ **yolov8m.pt** - لدقة أعلى
- ⭕ **yolov8l.pt** - لدقة أعلى (يحتاج GPU)
- ⭕ **yolov11x.pt** - أحدث نموذج (يحتاج GPU قوي)

---

## 🔄 التحميل التلقائي

### للنماذج القياسية من Ultralytics

النماذج التالية سيتم تنزيلها **تلقائياً** من Ultralytics عند أول استخدام:
- `yolov8n.pt`
- `yolov8s.pt`
- `yolov8m.pt`
- `yolov8l.pt`
- `yolov8x.pt`

```python
# سيتم التنزيل تلقائياً عند التشغيل
from ultralytics import YOLO
model = YOLO('yolov8s.pt')  # سيُحمّل تلقائياً إن لم يكن موجوداً
```

### للنموذج المخصص

⚠️ **النموذج المخصص `best.pt` يجب تحميله يدوياً** لأنه غير متاح على Ultralytics.

---

## 🚫 ما لا يجب فعله

❌ **لا ترفع النماذج إلى Git**
```bash
# ملف .gitignore يمنع ذلك بالفعل:
models/*.pt
```

❌ **لا تشارك النماذج المخصصة بدون إذن**

❌ **لا تحذف ملف `.gitkeep`** (يحافظ على المجلد في Git)

---

## 📐 مواصفات النماذج

### YOLOv8 Models

| النموذج | المعاملات | الحجم | mAP | السرعة (CPU) | السرعة (GPU) |
|---------|-----------|------|-----|--------------|--------------|
| YOLOv8n | 3.2M | 6.4MB | 37.3 | 80ms | 1.2ms |
| YOLOv8s | 11.2M | 22.5MB | 44.9 | 128ms | 1.9ms |
| YOLOv8m | 25.9M | 52MB | 50.2 | 234ms | 3.6ms |
| YOLOv8l | 43.7M | 87.7MB | 52.9 | 375ms | 5.7ms |
| YOLOv8x | 68.2M | 136.7MB | 53.9 | 479ms | 8.4ms |

### النموذج المخصص (best.pt)

- **الغرض**: تصنيف المرحلة الدراسية (High School / Middle School)
- **المعمارية**: مبني على YOLO
- **التدريب**: مدرب على dataset مخصص
- **الفئات**: High, Middle, students

---

## 🔧 استكشاف الأخطاء

### المشكلة: "Model not found"

```bash
# الحل: تأكد من وجود النموذج
ls -l models/*.pt

# إذا لم يكن موجوداً، حمّله من الرابط أعلاه
```

### المشكلة: "Permission denied"

```bash
# الحل: تعديل الصلاحيات
chmod 644 models/*.pt
```

### المشكلة: "Corrupted model file"

```bash
# الحل: أعد تحميل النموذج
rm models/yolov8s.pt
# ثم حمّل من جديد
```

### المشكلة: بطء التحميل

```bash
# نصيحة: استخدم download manager
# أو استخدم rclone لـ Google Drive:
pip install rclone
rclone copy gdrive:models/ models/
```

---

## 💾 تخزين النماذج

### مساحة التخزين المطلوبة

- **الحد الأدنى**: ~15MB (yolov8s فقط)
- **موصى به**: ~50MB (yolov8s + best.pt)
- **جميع النماذج**: ~300MB

### مكان التخزين

```
📁 models/
├── 📄 README.md (هذا الملف)
├── 📄 .gitkeep (للحفاظ على المجلد)
├── 🚫 yolov8s.pt (تحميل)
├── 🚫 yolov8m.pt (اختياري)
├── 🚫 yolov8l.pt (اختياري)
├── 🚫 yolov11x.pt (اختياري)
└── 🚫 best.pt (تحميل للنموذج المخصص)
```

🚫 = ممنوع من Git (.gitignore)

---

## 🆘 الدعم

إذا واجهت مشكلة في تحميل النماذج:

1. 📖 راجع [README الرئيسي](../README.md)
2. 💬 افتح [GitHub Discussion](https://github.com/aseelalmutari/StuTally-Project/discussions)
3. 🐛 أبلغ عن [Issue](https://github.com/aseelalmutari/StuTally-Project/issues)
4. 📧 راسلنا: support@stutally.project

---

## 📝 ملاحظات

- النماذج محمية بحقوق النشر لأصحابها (Ultralytics, Roboflow)
- استخدام النماذج يخضع لتراخيصها الأصلية
- للاستخدام التجاري، راجع تراخيص Ultralytics YOLO

---

<div align="center">

**بعد التحميل، أنت جاهز للبدء! 🚀**

[⬆️ العودة للأعلى](#-نماذج-yolo---دليل-التحميل)

</div>

---

**آخر تحديث**: أكتوبر 2025

