# 🤝 دليل المساهمة في StuTally

شكراً لاهتمامك بالمساهمة في مشروع StuTally! نرحب بجميع أنواع المساهمات سواء كانت إصلاح أخطاء، إضافة ميزات جديدة، تحسين التوثيق، أو حتى الإبلاغ عن المشاكل.

---

## 📋 جدول المحتويات

1. [قواعد السلوك](#قواعد-السلوك)
2. [كيف يمكنني المساهمة؟](#كيف-يمكنني-المساهمة)
3. [الإبلاغ عن الأخطاء](#الإبلاغ-عن-الأخطاء)
4. [اقتراح ميزات جديدة](#اقتراح-ميزات-جديدة)
5. [سير عمل المساهمة](#سير-عمل-المساهمة)
6. [معايير البرمجة](#معايير-البرمجة)
7. [كتابة الاختبارات](#كتابة-الاختبارات)
8. [توثيق الكود](#توثيق-الكود)

---

## 🌟 قواعد السلوك

### تعهدنا

نحن ملتزمون بتوفير بيئة ترحيبية وشاملة لجميع المساهمين. نتوقع من الجميع:

- استخدام لغة ترحيبية وشاملة
- احترام وجهات النظر والتجارب المختلفة
- قبول النقد البنّاء بلطف
- التركيز على ما هو أفضل للمجتمع
- إظهار التعاطف تجاه أعضاء المجتمع الآخرين

---

## 🎯 كيف يمكنني المساهمة؟

هناك عدة طرق للمساهمة في StuTally:

### 1. الإبلاغ عن الأخطاء 🐛

إذا وجدت خطأً:
- تأكد من أنه لم يتم الإبلاغ عنه مسبقاً في [Issues](https://github.com/aseelalmutari/StuTally-Project/issues)
- افتح Issue جديد مع وصف تفصيلي
- قدم معلومات كافية لإعادة إنتاج المشكلة

### 2. إصلاح الأخطاء 🔧

- تصفح [Issues](https://github.com/aseelalmutari/StuTally-Project/issues) للعثور على أخطاء للإصلاح
- الأخطاء المُعلمة بـ `good first issue` مناسبة للمبتدئين

### 3. إضافة ميزات جديدة ✨

- تحقق من [خارطة الطريق](README.md#-خارطة-الطريق-roadmap)
- اقترح ميزات جديدة عبر فتح Issue
- تأكد من أن الميزة تتماشى مع أهداف المشروع

### 4. تحسين التوثيق 📚

- إصلاح الأخطاء الإملائية أو النحوية
- إضافة أمثلة جديدة
- توضيح أجزاء غير واضحة
- ترجمة التوثيق إلى لغات أخرى

### 5. مراجعة الكود 👀

- راجع Pull Requests المفتوحة
- اختبر التغييرات محلياً
- قدم ملاحظات بنّاءة

---

## 🐛 الإبلاغ عن الأخطاء

### قبل الإبلاغ

1. **ابحث في Issues الموجودة**: تأكد من أن المشكلة لم يتم الإبلاغ عنها
2. **تحديث للأحدث**: تأكد من استخدامك أحدث إصدار
3. **تحقق من التوثيق**: قد يكون الحل موثقاً

### كيفية كتابة تقرير خطأ جيد

استخدم القالب التالي:

```markdown
## وصف المشكلة
[وصف واضح ومختصر للمشكلة]

## خطوات إعادة الإنتاج
1. اذهب إلى '...'
2. انقر على '...'
3. قم بـ '...'
4. شاهد الخطأ

## السلوك المتوقع
[ما كان يجب أن يحدث]

## السلوك الفعلي
[ما حدث فعلياً]

## لقطات الشاشة
[إن وُجدت]

## البيئة
- نظام التشغيل: [مثل: macOS 13.0, Windows 11, Ubuntu 22.04]
- إصدار Python: [مثل: 3.10.5]
- إصدار StuTally: [مثل: 2.0]
- CUDA/GPU: [نعم/لا، الإصدار]

## معلومات إضافية
[أي معلومات أخرى قد تكون مفيدة]

## السجلات (Logs)
```
[ألصق السجلات من logs/app.log]
```
```

---

## 💡 اقتراح ميزات جديدة

### قبل الاقتراح

1. تحقق من [خارطة الطريق](README.md#-خارطة-الطريق-roadmap)
2. ابحث في Issues للتأكد من عدم اقتراحها مسبقاً
3. فكر في التوافق مع أهداف المشروع

### كيفية كتابة اقتراح ميزة جيد

```markdown
## الملخص
[وصف مختصر للميزة المقترحة]

## الدافع
[لماذا هذه الميزة مفيدة؟]

## الحل المقترح
[كيف يجب أن تعمل الميزة؟]

## البدائل المعتبرة
[هل هناك طرق بديلة لتحقيق نفس الهدف؟]

## معلومات إضافية
[مخططات، أمثلة، روابط، إلخ]
```

---

## 🔄 سير عمل المساهمة

### 1. Fork المشروع

```bash
# انقر على زر "Fork" في GitHub
# ثم استنسخ نسختك
git clone https://github.com/YOUR_USERNAME/StuTally.git
cd StuTally
```

### 2. إعداد البيئة المحلية

```bash
# إنشاء بيئة افتراضية
python -m venv venv
source venv/bin/activate  # على Windows: venv\Scripts\activate

# تثبيت المتطلبات
pip install -r requirements.txt

# نسخ ملف البيئة
cp .env.example .env
# عدّل .env حسب الحاجة
```

### 3. إنشاء فرع جديد

```bash
# إنشاء فرع من main
git checkout -b feature/your-feature-name

# أو لإصلاح خطأ
git checkout -b fix/bug-description
```

**تسمية الفروع**:
- `feature/` - للميزات الجديدة
- `fix/` - لإصلاح الأخطاء
- `docs/` - لتحديثات التوثيق
- `refactor/` - لإعادة الهيكلة
- `test/` - لإضافة/تحديث الاختبارات

### 4. إجراء التعديلات

```bash
# قم بالتعديلات المطلوبة
# ...

# أضف الملفات المعدلة
git add .

# التزم بالتغييرات مع رسالة وصفية
git commit -m "Add: feature description"
```

**صيغة رسائل Commit**:
```
<type>: <description>

[optional body]

[optional footer]
```

**الأنواع المتاحة**:
- `Add`: إضافة ميزة جديدة
- `Fix`: إصلاح خطأ
- `Update`: تحديث ميزة موجودة
- `Refactor`: إعادة هيكلة الكود
- `Docs`: تحديث التوثيق
- `Test`: إضافة/تحديث اختبارات
- `Style`: تغييرات تنسيقية
- `Perf`: تحسين الأداء
- `Chore`: مهام صيانة

**أمثلة**:
```bash
git commit -m "Add: support for RTSP camera streaming"
git commit -m "Fix: video feed not loading on Safari browser"
git commit -m "Update: improve YOLO detection accuracy"
git commit -m "Docs: add API documentation for analytics endpoints"
```

### 5. اختبار التغييرات

```bash
# تشغيل الاختبارات
python -m pytest tests/

# تشغيل التطبيق للتأكد من عمله
python run.py
```

### 6. Push للمستودع

```bash
# رفع الفرع لـ fork الخاص بك
git push origin feature/your-feature-name
```

### 7. فتح Pull Request

1. اذهب إلى المستودع الأصلي على GitHub
2. انقر على "New Pull Request"
3. اختر فرعك
4. املأ قالب PR بالمعلومات المطلوبة
5. انتظر المراجعة

**قالب Pull Request**:
```markdown
## الوصف
[وصف مختصر للتغييرات]

## النوع
- [ ] إصلاح خطأ (Bug fix)
- [ ] ميزة جديدة (New feature)
- [ ] تحديث توثيق (Documentation update)
- [ ] إعادة هيكلة (Refactoring)
- [ ] آخر: ___________

## التغييرات
- تغيير 1
- تغيير 2
- ...

## الاختبار
[كيف تم اختبار التغييرات؟]

## لقطات الشاشة
[إن وُجدت]

## Checklist
- [ ] الكود يتبع أسلوب المشروع
- [ ] أضفت تعليقات، خاصة في المناطق المعقدة
- [ ] حدّثت التوثيق إن لزم الأمر
- [ ] لم أُضِف warnings جديدة
- [ ] أضفت اختبارات لإثبات أن الإصلاح/الميزة تعمل
- [ ] جميع الاختبارات الجديدة والقديمة تمر بنجاح
```

---

## 📝 معايير البرمجة

### أسلوب Python

نتبع [PEP 8](https://www.python.org/dev/peps/pep-0008/) مع بعض الاستثناءات:

```python
# ✅ جيد
def process_video_frame(frame, model_name, confidence_threshold=0.5):
    """
    Process a single video frame using YOLO detection.
    
    Args:
        frame: Input video frame
        model_name: Name of YOLO model to use
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Annotated frame with detections
    """
    # Implementation
    pass

# ❌ سيء
def ProcessVideoFrame(Frame,ModelName,confidence_threshold=0.5):
    # No docstring
    pass
```

### تنظيم الملفات

```python
# ترتيب Imports
# 1. مكتبات Python القياسية
import os
import sys
from datetime import datetime

# 2. مكتبات خارجية
import cv2
import numpy as np
from flask import Flask, render_template

# 3. مكتبات محلية
from app.services import video_service
from database import init_db
```

### التسمية

- **Classes**: `PascalCase` - مثل `VideoService`, `ModelLoader`
- **Functions/Methods**: `snake_case` - مثل `process_video()`, `get_analytics()`
- **Constants**: `UPPER_SNAKE_CASE` - مثل `MAX_FILE_SIZE`, `DEFAULT_MODEL`
- **Private**: `_leading_underscore` - مثل `_internal_method()`

### التعليقات والتوثيق

```python
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (tuple): First box coordinates (x1, y1, x2, y2)
        box2 (tuple): Second box coordinates (x1, y1, x2, y2)
        
    Returns:
        float: IoU value between 0 and 1
        
    Example:
        >>> box1 = (10, 10, 50, 50)
        >>> box2 = (30, 30, 70, 70)
        >>> iou = calculate_iou(box1, box2)
        >>> print(f"IoU: {iou:.2f}")
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Return implementation
    pass
```

### معالجة الأخطاء

```python
# ✅ جيد
try:
    result = process_video(video_path)
except FileNotFoundError as e:
    logger.error(f"Video file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error processing video: {e}", exc_info=True)
    return None

# ❌ سيء
try:
    result = process_video(video_path)
except:
    pass
```

---

## 🧪 كتابة الاختبارات

### بنية الاختبارات

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_models.py           # اختبارات النماذج
├── test_services.py         # اختبارات الخدمات
├── test_api.py              # اختبارات API
└── test_database.py         # اختبارات قاعدة البيانات
```

### مثال اختبار

```python
import pytest
from app.services.video_service import VideoService

class TestVideoService:
    """Test cases for VideoService"""
    
    def test_video_loading(self, sample_video_path):
        """Test that video loads successfully"""
        service = VideoService()
        result = service.load_video(sample_video_path)
        
        assert result is not None
        assert result['status'] == 'success'
        assert 'video_id' in result
    
    def test_video_processing_with_yolo(self, sample_video_path):
        """Test video processing with YOLO model"""
        service = VideoService()
        result = service.process_video(
            video_path=sample_video_path,
            model_name='yolov8s.pt'
        )
        
        assert result['detections_count'] > 0
        assert result['fps'] > 0
    
    def test_invalid_video_path(self):
        """Test handling of invalid video path"""
        service = VideoService()
        
        with pytest.raises(FileNotFoundError):
            service.load_video('nonexistent_video.mp4')
```

### تشغيل الاختبارات

```bash
# تشغيل جميع الاختبارات
pytest

# تشغيل ملف محدد
pytest tests/test_services.py

# تشغيل اختبار محدد
pytest tests/test_services.py::TestVideoService::test_video_loading

# مع تغطية الكود
pytest --cov=app tests/

# مع تقرير HTML
pytest --cov=app --cov-report=html tests/
```

---

## 📖 توثيق الكود

### Docstrings

استخدم صيغة Google Style:

```python
def get_analytics_data(video_id=None, time_range='all', filters=None):
    """
    Retrieve analytics data from database with optional filters.
    
    This function queries the detections database and aggregates
    statistics based on the provided parameters.
    
    Args:
        video_id (str, optional): Specific video ID to filter by.
            If None, includes all videos. Defaults to None.
        time_range (str, optional): Time range for data.
            Options: 'daily', 'weekly', 'monthly', 'all'.
            Defaults to 'all'.
        filters (dict, optional): Additional filters to apply.
            Keys can include 'stage', 'action', 'min_confidence'.
            Defaults to None.
    
    Returns:
        dict: Dictionary containing analytics data with keys:
            - 'total_students' (int): Total unique students detected
            - 'entries' (int): Number of entry events
            - 'exits' (int): Number of exit events
            - 'timestamps' (list): List of datetime strings
            - 'counts' (list): Corresponding counts
    
    Raises:
        ValueError: If time_range is not a valid option
        DatabaseError: If database connection fails
    
    Example:
        >>> data = get_analytics_data(
        ...     video_id='abc123',
        ...     time_range='daily',
        ...     filters={'stage': 'High School'}
        ... )
        >>> print(f"Total students: {data['total_students']}")
        Total students: 42
    
    Note:
        This function requires an active database connection.
        See `database.py` for connection setup.
    """
    # Implementation
    pass
```

### التعليقات في الكود

```python
# ✅ جيد - تعليق يشرح "لماذا"
# تخطي 3 إطارات لتحسين الأداء مع الحفاظ على دقة التتبع
if frame_count % 3 != 0:
    continue

# ✅ جيد - تعليق لشرح خوارزمية معقدة
# استخدام Kalman Filter للتنبؤ بموقع الكائن في الإطار التالي
predicted_position = kalman_filter.predict(current_position)

# ❌ سيء - تعليق يشرح "ماذا" (واضح من الكود)
# زيادة العداد
counter += 1
```

---

## 🎨 معايير الواجهة الأمامية

### HTML/CSS

```html
<!-- ✅ جيد -->
<div class="video-container">
    <video id="videoFeed" class="video-stream" autoplay></video>
    <div class="video-controls">
        <button id="playBtn" class="btn btn-primary">Play</button>
    </div>
</div>

<!-- ❌ سيء -->
<div style="width:100%;height:auto">
    <video id="v1" autoplay></video>
    <button onclick="play()">Play</button>
</div>
```

### JavaScript

```javascript
// ✅ جيد
class VideoPlayer {
    constructor(videoElement) {
        this.video = videoElement;
        this.isPlaying = false;
        this.init();
    }
    
    init() {
        this.attachEventListeners();
    }
    
    attachEventListeners() {
        this.video.addEventListener('play', () => {
            this.isPlaying = true;
            this.updateUI();
        });
    }
}

// ❌ سيء
function play() {
    var v = document.getElementById('v1');
    v.play();
}
```

---

## 🏷️ إصدارات البرنامج (Versioning)

نتبع [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

مثال: 2.1.3
```

- **MAJOR**: تغييرات غير متوافقة مع الإصدارات السابقة
- **MINOR**: إضافة ميزات جديدة متوافقة
- **PATCH**: إصلاحات أخطاء متوافقة

---

## 📦 إصدار نسخة جديدة

### قائمة التحقق (للمشرفين فقط)

1. تحديث رقم الإصدار في `version.py`
2. تحديث `CHANGELOG.md`
3. تشغيل جميع الاختبارات
4. عمل commit للتغييرات
5. إنشاء tag للإصدار
6. Push للمستودع مع tags

```bash
# تحديث الإصدار
echo "__version__ = '2.1.0'" > app/version.py

# Commit
git add .
git commit -m "Release: version 2.1.0"

# إنشاء Tag
git tag -a v2.1.0 -m "Version 2.1.0 - Feature updates"

# Push
git push origin main --tags
```

---

## ❓ أسئلة؟

إذا كان لديك أي أسئلة حول المساهمة:

- 💬 افتح [Discussion](https://github.com/aseelalmutari/StuTally-Project/discussions)
- 📧 راسلنا على: your.email@example.com
- 📖 راجع [التوثيق](README.md)

---

## 🙏 شكر خاص

شكراً لجميع [المساهمين](https://github.com/aseelalmutari/StuTally-Project/graphs/contributors) الذين ساعدوا في تطوير StuTally!

---

<div align="center">

**صُنع بـ ❤️ بواسطة مجتمع StuTally**

[⬆️ العودة للأعلى](#-دليل-المساهمة-في-stutally)

</div>

