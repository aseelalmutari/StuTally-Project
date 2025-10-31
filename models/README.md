# 📦 YOLO Models - Download Guide

This folder contains YOLO models used in StuTally for person detection and classification.

---

## 🔗 Download Models

### ⚠️ Important Note

Due to large file sizes of YOLO models (**from 6MB to 400MB+**), they are not included in the GitHub repository.

### 📥 Download Links

#### 🌐 Main Link (Recommended)

**[📥 Download All Models - Google Drive](https://drive.google.com/drive/folders/1FbXiXZtd6Zf8A_8d77Tz5kswGu5m-whX?usp=share_link)**

Or

**[📥 Download Models - Dropbox](dropbox_link_here)**

Or

**[📥 Download Models - OneDrive](onedrive_link_here)**

#### 🔗 Individual Model Direct Links

If you want to download a specific model only:

| Model | Size | Description | Download Link |
|-------|------|-------------|---------------|
| `yolov8s.pt` | ~11MB | Default model (recommended) | [Download](direct_link) |
| `yolov8m.pt` | ~26MB | Medium model (higher accuracy) | [Download](direct_link) |
| `yolov8l.pt` | ~44MB | Large model (high accuracy) | [Download](direct_link) |
| `yolov11x.pt` | ~136MB | Very large v11 model | [Download](direct_link) |
| `best.pt` | ~size_tbd~ | **Custom Model** (stage classification) | [Download](direct_link) |

---

## 📂 Installation

### Method 1: Manual Download

1. **Download models** from the links above
2. **Extract** (if compressed)
3. **Move files** to `models/` folder:

```bash
# From download folder
cp ~/Downloads/yolov8s.pt /Users/youuser/Desktop/StuTally/models/
cp ~/Downloads/best.pt /Users/youuser/Desktop/StuTally/models/

# Or using relative path
cd /Users/youuser/Desktop/StuTally
cp ~/Downloads/*.pt models/
```

### Method 2: Using wget/curl (if you have a direct link)

```bash
cd models/

# Using wget
wget -O yolov8s.pt "download_link"

# Or using curl
curl -L -o yolov8s.pt "download_link"
```

### Method 3: Using gdown (for Google Drive downloads)

```bash
# Install gdown
pip install gdown

# Download from Google Drive
cd models/
gdown "google_drive_file_id"

# Or download entire folder
gdown --folder "google_drive_folder_id"
```

---

## ✅ Verify Installation

After downloading, make sure models exist:

```bash
cd models/
ls -lh

# You should see:
# -rw-r--r--  1 user  staff   11M  yolov8s.pt
# -rw-r--r--  1 user  staff   26M  yolov8m.pt
# -rw-r--r--  1 user  staff   44M  yolov8l.pt
# -rw-r--r--  1 user  staff  136M  yolov11x.pt
# -rw-r--r--  1 user  staff   XXM  best.pt
```

Or using Python:

```python
import os
from pathlib import Path

models_dir = Path(__file__).parent
models = list(models_dir.glob('*.pt'))

print("Installed models:")
for model in models:
    size_mb = model.stat().st_size / (1024 * 1024)
    print(f"  ✓ {model.name} ({size_mb:.1f} MB)")
```

---

## 📋 Required Models

### Essential (Minimum)

- ✅ **yolov8s.pt** - Default model (required)

### Recommended

- ✅ **yolov8s.pt** - For general use
- ✅ **best.pt** - For custom classification (academic stage classification)

### Optional

- ⭕ **yolov8m.pt** - For higher accuracy
- ⭕ **yolov8l.pt** - For higher accuracy (needs GPU)
- ⭕ **yolov11x.pt** - Latest model (needs powerful GPU)

---

## 🔄 Auto-Download

### For Standard Ultralytics Models

The following models will be downloaded **automatically** from Ultralytics on first use:
- `yolov8n.pt`
- `yolov8s.pt`
- `yolov8m.pt`
- `yolov8l.pt`
- `yolov8x.pt`

```python
# Will be downloaded automatically on first run
from ultralytics import YOLO
model = YOLO('yolov8s.pt')  # Will download automatically if not found
```

### For Custom Model

⚠️ **Custom model `best.pt` must be downloaded manually** as it's not available on Ultralytics.

---

## 🚫 What NOT to Do

❌ **Do not upload models to Git**
```bash
# .gitignore already prevents this:
models/*.pt
```

❌ **Do not share custom models without permission**

❌ **Do not delete `.gitkeep` file** (keeps folder in Git)

---

## 📐 Model Specifications

### YOLOv8 Models

| Model | Parameters | Size | mAP | Speed (CPU) | Speed (GPU) |
|-------|-----------|------|-----|-------------|-------------|
| YOLOv8n | 3.2M | 6.4MB | 37.3 | 80ms | 1.2ms |
| YOLOv8s | 11.2M | 22.5MB | 44.9 | 128ms | 1.9ms |
| YOLOv8m | 25.9M | 52MB | 50.2 | 234ms | 3.6ms |
| YOLOv8l | 43.7M | 87.7MB | 52.9 | 375ms | 5.7ms |
| YOLOv8x | 68.2M | 136.7MB | 53.9 | 479ms | 8.4ms |

### Custom Model (best.pt)

- **Purpose**: Academic stage classification (High School / Middle School)
- **Architecture**: Built on YOLO
- **Training**: Trained on custom dataset
- **Classes**: High, Middle, students

---

## 🔧 Troubleshooting

### Issue: "Model not found"

```bash
# Solution: Verify model exists
ls -l models/*.pt

# If not found, download from link above
```

### Issue: "Permission denied"

```bash
# Solution: Fix permissions
chmod 644 models/*.pt
```

### Issue: "Corrupted model file"

```bash
# Solution: Re-download model
rm models/yolov8s.pt
# Then download again
```

### Issue: Slow Download

```bash
# Tip: Use download manager
# Or use rclone for Google Drive:
pip install rclone
rclone copy gdrive:models/ models/
```

---

## 💾 Model Storage

### Required Storage Space

- **Minimum**: ~15MB (yolov8s only)
- **Recommended**: ~50MB (yolov8s + best.pt)
- **All Models**: ~300MB

### Storage Location

```
📁 models/
├── 📄 README.md (this file)
├── 📄 .gitkeep (keeps folder in Git)
├── 🚫 yolov8s.pt (download)
├── 🚫 yolov8m.pt (optional)
├── 🚫 yolov8l.pt (optional)
├── 🚫 yolov11x.pt (optional)
└── 🚫 best.pt (download for custom model)
```

🚫 = Excluded from Git (.gitignore)

---

## 🆘 Support

If you encounter issues downloading models:

1. 📖 Check [Main README](../README.md)
2. 💬 Open [GitHub Discussion](https://github.com/aseelalmutari/StuTally-Project/discussions)
3. 🐛 Report [Issue](https://github.com/aseelalmutari/StuTally-Project/issues)
4. 📧 Email us: caa73061@gmail.com

---

## 📝 Notes

- Models are copyrighted by their owners (Ultralytics, Roboflow)
- Model usage subject to their original licenses
- For commercial use, review Ultralytics YOLO licenses

---

<div align="center">

**After download, you're ready to start! 🚀**

[⬆️ Back to Top](#-yolo-models---download-guide)

</div>

---

**Last Updated**: October 2025

