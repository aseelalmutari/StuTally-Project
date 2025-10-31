<div align="center">

# 🎓 StuTally

### Advanced Intelligent System for Student Counting and Tracking

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An integrated system for student counting and real-time movement analysis using artificial intelligence, featuring an interactive analytics dashboard and advanced YOLO models.

[Features](#-key-features) • [Installation](#-installation--setup) • [Usage](#-usage) • [Documentation](#-project-structure)

</div>

---

## 📋 Overview

**StuTally** is a comprehensive intelligent system for student counting and tracking using computer vision and AI technologies. The system provides advanced capabilities for:

- 🎯 **Accurate Detection**: Using YOLO models (v8/v11) to detect persons in images and videos
- 🔍 **Smart Tracking**: Advanced DeepSort technology for tracking individuals and counting entries/exits
- 🎓 **Academic Stage Classification**: Custom model for classifying students by educational stage (High School/Middle School)
- 📊 **Comprehensive Analytics**: Interactive dashboard with KPIs, charts, heatmaps, and comparisons
- 🔒 **Integrated Security**: User authentication with Flask-Login and JWT
- 📤 **Data Export**: Download reports in CSV and PDF formats
- ⚡ **Real-time Processing**: Live video streaming via WebSocket

---

## ✨ Key Features

### 🤖 AI & Detection

- Multiple YOLO model support (n, s, m, l, x) with auto-download capability
- High-accuracy person detection in videos and images
- Custom model for educational stage classification (High School / Middle School) via Roboflow
- Smart object tracking using DeepSort
- GPU/CUDA support for accelerated processing
- Performance optimization with frame skipping

### 📊 Analytics & Statistics

- **Key Performance Indicators (KPIs)**:
  - Total student count
  - Total entries
  - Total exits
  
- **Interactive Charts**:
  - Students Over Time: changes in numbers over time
  - Heatmap: activity map by day and hour
  - Stage Comparison: comparison between educational stages
  
- **Advanced Filtering**:
  - By video/image
  - By time range (daily/weekly/monthly)
  - By specific date

### 🔄 Real-time Processing

- Live video streaming via MJPEG
- Real-time updates via WebSocket (Flask-SocketIO)
- Optional counting line
- Automatic entry/exit calculation when crossing the line

### 🔐 Security & Authentication

- Integrated login system (Flask-Login)
- Password encryption (Bcrypt)
- JWT authentication for APIs
- Protected analytics pages
- Session and role management

---

## 🛠️ Requirements

### Basic Requirements

- **Python**: 3.8 or later
- **Operating System**: macOS, Windows, or Linux
- **Optional**: CUDA/GPU for accelerated YOLO processing

### Core Libraries

#### Backend Framework
```
Flask==3.0.0
Flask-SocketIO==5.3.5
Flask-Login==0.6.3
Flask-Bcrypt==1.0.1
Flask-JWT-Extended==4.6.0
eventlet==0.33.3
```

#### Computer Vision & AI
```
ultralytics>=8.0.0
opencv-python==4.8.1.78
torch>=2.0.0
torchvision>=0.15.0
inference>=0.9.0
supervision
```

#### Object Tracking
```
filterpy==1.4.5
scikit-learn>=1.3.0
scipy>=1.11.0
```

#### Data Processing & Export
```
pandas>=2.1.0
numpy>=1.24.0,<2.0
reportlab==4.0.7
plotly>=5.18.0
matplotlib>=3.8.0
```

#### Utilities
```
APScheduler>=3.10.4
python-dotenv==1.0.0
Pillow>=10.0.0
requests>=2.31.0
```

---

## 📦 Installation & Setup

### 1️⃣ Clone the Project

```bash
# Clone the repository
git clone https://github.com/aseelalmutari/StuTally-Project.git
cd StuTally
```

### 2️⃣ Create Virtual Environment

#### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Using Conda (Optional)
```bash
# Create environment from conda file
conda env create -f env_StuTally.yml

# Activate environment
conda activate t6
```

### 3️⃣ Install Requirements

```bash
# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

```bash
# Copy environment template file
cp .env.example .env

# Edit .env file with text editor
nano .env
```

#### `.env` file contents:
```bash
# Flask Configuration
SECRET_KEY=your-super-secret-key-change-this
DEBUG=True
FLASK_ENV=development

# JWT Configuration
JWT_SECRET_KEY=your-jwt-secret-key-change-this
JWT_ACCESS_TOKEN_EXPIRES=3600

# Roboflow API (for custom model)
ROBOFLOW_API_KEY=your-roboflow-api-key
ROBOFLOW_MODEL_ID=ss-uniform/3

# Model Configuration
DEFAULT_MODEL=yolov8s.pt
FRAME_SKIP=3
DEVICE=auto

# Server Configuration
HOST=0.0.0.0
PORT=5000

# Database
DATABASE_TYPE=sqlite
SQLITE_PATH=data/detections.db
```

### 5️⃣ Create Required Folders

```bash
# Folders will be created automatically on startup, or create manually:
mkdir -p uploads models data logs static/processed_images
```

### 6️⃣ Initialize Database and Create User

```bash
# Create database and admin user
python create_user.py
```

> **Note**: A default user will be created:
> - **Username**: `admin`
> - **Password**: `admin123`
> 
> ⚠️ **Important**: Change the password after first login!

### 7️⃣ Download YOLO Models

#### 📥 Download Models from External Link

> 📦 **Important Note**: Due to large file sizes of YOLO models (ranging from 6MB to 400MB+), they are not included in the repository.
> 
> 🔗 **You can download the trained models from the following link**:
> 
> **[📥 Download Models - Google Drive](https://drive.google.com/drive/folders/1FbXiXZtd6Zf8A_8d77Tz5kswGu5m-whX?usp=share_link)**

#### Manual Model Installation

After downloading, place `.pt` files in the `models/` folder:

```bash
# Create models folder if it doesn't exist
mkdir -p models

# Move downloaded models
# Example:
models/
├── yolov8s.pt          # Default model (11MB)
├── yolov8m.pt          # Medium model (26MB)
├── yolov8l.pt          # Large model (44MB)
├── yolo11x.pt          # Very large v11 model (136MB)
└── best.pt             # Custom model (stage classification)
```

#### Auto-Download (Standard Models Only)

Standard models from Ultralytics will be downloaded automatically on first use:
- `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`

⚠️ **Note**: The custom model `best.pt` must be downloaded manually from the link above.

---

## 🚀 Running the Application

### Local Development

```bash
# Method 1: Using app.py
python app.py

# Method 2: Using run.py (updated architecture)
python run.py
```

The server will run on: `http://localhost:5000`

### Production Deployment

```bash
# Set environment to production
export FLASK_ENV=production

# Use Gunicorn with eventlet
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 run:app
```

### Verify Setup

Open your browser and navigate to:
- **Homepage**: http://localhost:5000
- **Login Page**: http://localhost:5000/login
- **Health Check**: http://localhost:5000/health

---

## 💡 Usage

### 1. Main Interface (Upload & Processing)

#### Upload and Process Video

1. Open the homepage: `http://localhost:5000`
2. Select a video file (MP4, AVI, MOV, MKV)
3. Choose model from dropdown:
   - **YOLOv8/v11**: General person detection
   - **best.pt**: Custom model (academic stage classification)
4. Enable "Enable Counting Line" for entry/exit counting (optional)
5. Click "Upload"
6. Watch live streaming of processed video

#### Process Image

1. Select an image file (JPG, PNG, BMP, GIF)
2. Choose appropriate model
3. Click "Upload"
4. Processed image with detection results will appear

### 2. Analytics Dashboard

#### Login

```bash
# Navigate to login page
http://localhost:5000/login

# Use default credentials:
Username: admin
Password: admin123
```

#### Explore Analytics

After logging in, you can:

- **View KPIs**: Total students, entries, exits
- **Charts**:
  - Students Over Time: changes in numbers over time
  - Heatmap: activity map (day/hour)
  - Stage Comparison: academic stage comparison
  
- **Filtering**:
  - By specific video/image
  - By time period
  - By specific date

- **Export Data**:
  - Download CSV: `Download CSV`
  - Download PDF: `Download PDF`

### 3. Using the API

#### Get JWT Token

```bash
curl -X POST http://localhost:5000/api/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin123"
  }'
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

#### Access Analytics via API

```bash
curl -X GET http://localhost:5000/api/analytics \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

#### Available API Routes

| Route | Method | Description | Authentication |
|-------|--------|-------------|----------------|
| `/api/login` | POST | Get JWT Token | No |
| `/api/analytics` | GET | Get Analytics | JWT ✅ |
| `/analytics/kpis` | GET | Get KPIs | Login ✅ |
| `/analytics/students_over_time` | GET | Students Over Time Data | Login ✅ |
| `/analytics/heatmap_data` | GET | Heatmap Data | Login ✅ |
| `/analytics/stage_comparison_data` | GET | Stage Comparison | Login ✅ |
| `/videos` | GET | List Uploaded Videos | Login ✅ |
| `/analytics/download` | GET | Download CSV | Login ✅ |
| `/analytics/download_pdf` | GET | Download PDF | Login ✅ |

---

## 📁 Project Structure

```
StuTally/
│
├── 📄 app.py                     # Main Application (Legacy)
├── 📄 run.py                     # Entry Point (Updated)
├── 📄 auth.py                    # Authentication & Users
├── 📄 database.py                # SQLite Database Management
├── 📄 create_user.py             # Admin User Creation Script
├── 📄 yolo.py                    # Experimental YOLO Code
├── 📄 realtime.py                # Real-time Integrations
├── 📄 requirements.txt           # Project Dependencies
├── 📄 env_StuTally.yml           # Conda Environment File
├── 📄 README.md                  # Main Documentation
├── 📄 .gitignore                 # Git Ignore File
│
├── 📂 app_modules/               # Main Application Package
│   ├── __init__.py              # Application Factory
│   ├── extensions.py            # Extension Setup
│   │
│   ├── 📂 routes/               # Flask Routes
│   │   ├── __init__.py
│   │   ├── main.py             # Main Routes (Upload/Stream)
│   │   ├── analytics.py        # Analytics Routes
│   │   └── api.py              # API Endpoints
│   │
│   ├── 📂 services/             # Business Services
│   │   ├── __init__.py
│   │   ├── upload_service.py   # File Upload Processing
│   │   ├── video_service.py    # Video Processing & Streaming
│   │   ├── image_service.py    # Image Processing
│   │   ├── analytics_service.py # Analytics Logic
│   │   ├── export_service.py   # CSV/PDF Export
│   │   └── realtime_service.py # Real-time Service
│   │
│   └── 📂 utils/                # Helper Utilities
│       └── __init__.py
│
├── 📂 ml/                        # Machine Learning Modules
│   ├── __init__.py
│   ├── model_loader.py          # Model Loading & Management
│   └── model_registry.py        # Model Registry
│
├── 📂 config/                    # Application Configuration
│   ├── __init__.py
│   └── config.py                # Main Configuration File
│
├── 📂 templates/                 # HTML Templates
│   ├── index.html               # Homepage
│   ├── analytics.html           # Analytics Dashboard
│   ├── login.html               # Login Page
│   └── index_test.html          # Test Page
│
├── 📂 static/                    # Static Files
│   ├── index.css
│   ├── base.css
│   ├── analytics.css
│   ├── 📂 fonts/
│   │   ├── Nexa-ExtraLight.ttf
│   │   └── Nexa-Heavy.ttf
│   └── 📂 processed_images/     # Processed Images
│
├── 📂 models/                    # YOLO Models (.pt) ⚠️ Not in Git
│   ├── README.md                # 📥 Model Download Guide
│   ├── .gitkeep                 # Keep Folder in Git
│   ├── yolov8s.pt              # 🔗 Download from external link
│   ├── yolov8m.pt              # 🔗 Download from external link
│   ├── yolo11x.pt              # 🔗 Download from external link
│   └── best.pt                  # 🔗 Custom Model (Manual Download)
│
├── 📂 uploads/                   # Uploaded Files
│
├── 📂 data/                      # Database
│   └── detections.db            # SQLite Database
│
├── 📂 logs/                      # Application Logs
│   └── app.log
│
└── 📂 scripts/                   # Helper Scripts
    ├── generate_secrets.py      # Secret Key Generation
    ├── setup_env.sh             # Environment Setup
    ├── test_full_system.py      # Full System Test
    └── test_week1.py            # Week 1 Test
```

---

## 🎯 How the System Works

### Workflow

```
1. File Upload (Video/Image)
         ↓
2. Model Selection (YOLO / Custom)
         ↓
3. Process File
   ├─ Video: YOLO + DeepSort
   │   ├─ Person Detection
   │   ├─ Movement Tracking
   │   ├─ Entry/Exit Counting
   │   └─ Stream Processed Video
   │
   └─ Image: YOLO / Custom Model
       ├─ Detection & Classification
       └─ Save Processed Image
         ↓
4. Store Data in Database
         ↓
5. Display Results & Analytics
   ├─ Analytics Dashboard
   ├─ Charts
   └─ Export Reports
```

### Technologies Used

#### Backend
- **Flask**: Web Framework
- **Flask-SocketIO**: Real-time bidirectional communication
- **SQLite**: Lightweight Database
- **Threading**: Non-blocking WSGI Server

#### AI & Computer Vision
- **Ultralytics YOLO**: Object Detection
- **DeepSort**: Multi-object Tracking
- **OpenCV**: Image & Video Processing
- **PyTorch**: Deep Learning Engine
- **Roboflow Inference**: Custom Model

#### Frontend
- **HTML5/CSS3**: User Interface
- **JavaScript**: Page Interactivity
- **Socket.IO Client**: Real-time Connection

#### Security
- **Flask-Login**: Session Management
- **Flask-Bcrypt**: Password Encryption
- **Flask-JWT-Extended**: JWT Authentication

---

## 🔧 Configuration & Customization

### Change Default Model

In `.env` file:
```bash
DEFAULT_MODEL=yolov8m.pt  # or yolov8l.pt, yolo11x.pt, etc.
```

### Frame Skipping (Performance Optimization)

```bash
FRAME_SKIP=5  # Process every 5 frames only
```

### DeepSort Settings

In `config/config.py`:
```python
DEEPSORT_MAX_AGE = 15        # Maximum track age
DEEPSORT_N_INIT = 3          # Number of frames for confirmation
DEEPSORT_MAX_IOU = 0.7       # IoU threshold
```

### Device Selection (CPU/GPU)

```bash
DEVICE=cuda   # Use GPU
DEVICE=cpu    # Use CPU
DEVICE=auto   # Auto-select
```

---

## 📊 Database

### Main Tables

#### 1. `detections`
Stores all detection events:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Unique ID |
| video_id | TEXT | Video/Image ID |
| class_name | TEXT | Class Name (person, High, Middle) |
| confidence | REAL | Confidence Score |
| x1, y1, x2, y2 | INTEGER | Bounding Box Coordinates |
| track_id | INTEGER | Tracking ID |
| stage | TEXT | Academic Stage |
| datetime | TEXT | Detection Time |
| action | TEXT | Action (entry/exit/counted) |
| media_type | TEXT | Media Type (video/image) |

#### 2. `videos`
Uploaded file information:

| Column | Type | Description |
|--------|------|-------------|
| video_id | TEXT | Unique ID (PRIMARY KEY) |
| video_path | TEXT | File Path |
| counting_line_enabled | BOOLEAN | Counting Line Enabled |
| datetime | TEXT | Upload Time |

#### 3. `users`
Users and Authentication:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Unique ID |
| username | TEXT | Username |
| password | TEXT | Password (Encrypted) |
| role | TEXT | Role (admin/user) |

#### 4. `daily_statistics`
Daily Statistics:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Unique ID |
| date | TEXT | Date |
| class_name | TEXT | Class Name |
| stage | TEXT | Academic Stage |
| count | INTEGER | Total Count |
| entries | INTEGER | Entry Count |
| exits | INTEGER | Exit Count |

---

## 🧪 Testing & Development

### Run Tests

```bash
# Full system test
python scripts/test_full_system.py

# Specific test
python scripts/test_week1.py
```

### Run in Development Mode

```bash
# Enable DEBUG mode
export DEBUG=True
export FLASK_ENV=development

python run.py
```

---

## 🐛 Troubleshooting

### Issue: "Model not found" or model not loading

**Cause**: Models are not included in the repository due to large file sizes.

**Solution**:
1. **Download models from external link**:
   ```bash
   # See models/README.md for links
   # Or use the direct link from "Download YOLO Models" section above
   ```

2. **Place models in correct folder**:
   ```bash
   cp ~/Downloads/yolov8s.pt models/
   ls -lh models/*.pt  # Verify
   ```

3. **For standard models - auto-download**:
   - Ensure internet connection
   - Models will be downloaded automatically from Ultralytics

4. **Check logs**:
   ```bash
   tail -f logs/app.log
   ```

📖 **For more details**: See [models/README.md](models/README.md)

### Issue: Slow Processing

**Solution**:
1. Increase `FRAME_SKIP` value in `.env`
2. Use smaller model (yolov8n.pt)
3. Enable GPU/CUDA if available

### Issue: Database Connection Error

**Solution**:
```bash
# Reinitialize database
rm data/detections.db
python create_user.py
```

### Issue: Custom Model Not Working

**Solution**:
1. Ensure `ROBOFLOW_API_KEY` is set in `.env`
2. Verify `ROBOFLOW_MODEL_ID` is correct
3. Confirm `inference` library is installed

---

## 🤝 Contributing

We welcome contributions from the community! Here's how to contribute:

### Contribution Steps

1. **Fork the Project**
```bash
git clone https://github.com/aseelalmutari/StuTally-Project.git
```

2. **Create a New Branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make Changes**
```bash
# Make your changes
git add .
git commit -m "Add amazing feature"
```

4. **Push to Repository**
```bash
git push origin feature/amazing-feature
```

5. **Open Pull Request**
   - Go to project page on GitHub
   - Open new Pull Request
   - Clearly describe changes

### Contribution Guidelines

- 📝 Write clear, commented code
- ✅ Add tests for new features
- 📚 Update documentation as needed
- 🎨 Follow the coding style
- 🐛 Report bugs via Issues

### Contribution Areas

- ✨ Add new features
- 🐛 Fix bugs
- 📝 Improve documentation
- 🎨 Enhance UI
- ⚡ Optimize performance
- 🌐 Translate to other languages

---

## 📜 License

This project is licensed under **MIT License** - see [LICENSE](LICENSE) file for details.

### Important License Notes

⚠️ **Used Libraries**:
- **Ultralytics YOLO**: [AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)
- **DeepSort**: Check library license
- **OpenCV**: [Apache 2.0 License](https://opencv.org/license/)
- **PyTorch**: [BSD License](https://github.com/pytorch/pytorch/blob/master/LICENSE)

For commercial use, please review all library licenses used.

---

## 👥 Team & Support

### Developers

- **Lead Developer**: [ASEEL](https://github.com/aseelalmutari)
- **Contributors**: [Contributors List](https://github.com/aseelalmutari/StuTally-Project/graphs/contributors)

### Contact & Support

- 📧 **Email**: caa73061@gmail.com
- 🐛 **Report Issues**: [GitHub Issues](https://github.com/aseelalmutari/StuTally-Project/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/aseelalmutari/StuTally-Project/discussions)
- 📖 **Full Documentation**: [Wiki](https://github.com/aseelalmutari/StuTally-Project/wiki)

### Financial Support

If you're using StuTally and want to support the project:

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow?style=for-the-badge&logo=buy-me-a-coffee)](https://buymeacoffee.com/yourusername)

---

## 🗺️ Roadmap

### Upcoming Releases

#### v3.0 (Planned)
- [ ] PostgreSQL support for scalability
- [ ] Advanced admin interface
- [ ] Multi-level permission system
- [ ] Live RTSP camera support
- [ ] Real-time notifications
- [ ] External system integration (API Webhooks)

#### v2.1 (Coming Soon)
- [ ] Processing performance improvements
- [ ] Latest YOLO model support
- [ ] Multi-language interface (i18n)
- [ ] Excel export
- [ ] Additional charts

#### v2.0 (Current)
- [x] Improved architecture with Flask Factory
- [x] Modular service system
- [x] Model management system
- [x] Security improvements
- [x] Comprehensive documentation

---

## 📚 Additional Resources

### Documentation

- [Full User Guide](docs/user-guide.md) (Coming Soon)
- [Developer Guide](docs/developer-guide.md) (Coming Soon)
- [API Reference](docs/api-reference.md) (Coming Soon)

### Video Tutorials

- [Installation Guide](https://youtube.com/...) (Coming Soon)
- [Usage Guide](https://youtube.com/...) (Coming Soon)
- [Customization & Configuration](https://youtube.com/...) (Coming Soon)

### Learning Resources

- [YOLO Documentation](https://docs.ultralytics.com/)
- [DeepSort Paper](https://arxiv.org/abs/1703.07402)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

---

## 🌟 Acknowledgments

We thank the following projects that contributed to StuTally's success:

- **Ultralytics** - Amazing YOLO Models
- **Flask** - Simple and powerful framework
- **OpenCV** - Core computer vision library
- **PyTorch** - Deep learning engine
- **DeepSort** - Advanced tracking algorithm

---

## 📸 Screenshots

<div align="center">

### Main Interface
[![Main Interface](https://drive.google.com/uc?export=view&id=1KChL-vsQAqdhzJ14feIp_diBog8CKJF8)](https://drive.google.com/file/d/1KChL-vsQAqdhzJ14feIp_diBog8CKJF8/view?usp=sharing)

### Analytics Dashboard
**Main Analytics Dashboard - KPIs & Statistics:**
[![Analytics Dashboard - KPIs](https://drive.google.com/uc?export=view&id=10ZMFpAUWcQwebvb-8OIp3bE3ypOnihvz)](https://drive.google.com/file/d/10ZMFpAUWcQwebvb-8OIp3bE3ypOnihvz/view?usp=share_link)

**Time Series Charts:**
[![Analytics Dashboard - Charts](https://drive.google.com/uc?export=view&id=1esdkcef3X3ayC9UO85pYzIDRo6tVlMqy)](https://drive.google.com/file/d/1esdkcef3X3ayC9UO85pYzIDRo6tVlMqy/view?usp=share_link)

**Heatmap:**
[![Analytics Dashboard - Heatmap](https://drive.google.com/uc?export=view&id=1vx3uZBMsSF5syPJN8qGlZ5nqmasZJIFX)](https://drive.google.com/file/d/1vx3uZBMsSF5syPJN8qGlZ5nqmasZJIFX/view?usp=share_link)

**Stage Comparison:**
[![Analytics Dashboard - Stage Comparison](https://drive.google.com/uc?export=view&id=1egSApf2gzd16fokh2nsp5zLI_zUSnCS7)](https://drive.google.com/file/d/1egSApf2gzd16fokh2nsp5zLI_zUSnCS7/view?usp=share_link)

### Academic Stage Processing
**Stage Classification - Image 1:**
[![Academic Stage Classification 1](https://drive.google.com/uc?export=view&id=1KlK0ziD5DttCihxO5thwu_sQVkpetGMz)](https://drive.google.com/file/d/1KlK0ziD5DttCihxO5thwu_sQVkpetGMz/view?usp=share_link)

**Stage Classification - Image 2:**
[![Academic Stage Classification 2](https://drive.google.com/uc?export=view&id=1qxfwoKIyUwhNTwOs-4IjF1yrCFs-Yd64)](https://drive.google.com/file/d/1qxfwoKIyUwhNTwOs-4IjF1yrCFs-Yd64/view?usp=share_link)

</div>

---

## ⚡ Performance & Specifications

### Processing Rates

| Model | Resolution | FPS (CPU) | FPS (GPU) | Memory |
|-------|-----------|-----------|-----------|---------|
| YOLOv8n | 640x640 | 12-15 | 60-80 | ~2GB |
| YOLOv8s | 640x640 | 8-12 | 45-60 | ~3GB |
| YOLOv8m | 640x640 | 5-8 | 30-45 | ~4GB |
| YOLOv8l | 640x640 | 3-5 | 20-30 | ~6GB |

### System Requirements

#### Minimum
- CPU: Intel Core i5 or equivalent
- RAM: 8GB
- Storage: 5GB available
- OS: Windows 10 / macOS 10.14 / Ubuntu 18.04+

#### Recommended
- CPU: Intel Core i7 or later
- RAM: 16GB+
- GPU: NVIDIA GTX 1060 or better
- CUDA: 11.0+
- Storage: 20GB+ SSD

---

## ❓ FAQ

<details>
<summary><b>Can I use StuTally without GPU?</b></summary>

Yes, the system works on CPU but with lower performance. Use smaller models (yolov8n) and increase FRAME_SKIP value to improve performance.
</details>

<details>
<summary><b>What video and image formats are supported?</b></summary>

- **Video**: MP4, AVI, MOV, MKV
- **Images**: JPG, JPEG, PNG, BMP, GIF
</details>

<details>
<summary><b>How do I add a custom YOLO model?</b></summary>

1. Place `.pt` file in `models/` folder
2. Add model name to `YOLO_MODELS` in `config/config.py`
3. Restart application
</details>

<details>
<summary><b>Does the system support live cameras?</b></summary>

Currently, the system supports file uploads only. Live camera support (RTSP) is planned for v3.0.
</details>

<details>
<summary><b>How do I change user password?</b></summary>

```python
# In Python shell
from auth import bcrypt
new_password_hash = bcrypt.generate_password_hash('new_password').decode('utf-8')
# Then manually update database
```
</details>

---

<div align="center">

## 🎉 Thank You for Using StuTally!

If you like the project, don't forget to add ⭐ to the repository!

**Made with ❤️ using Python & YOLO**

---

[![GitHub Stars](https://img.shields.io/github/stars/aseelalmutari/StuTally-Project?style=social)](https://github.com/aseelalmutari/StuTally-Project/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/aseelalmutari/StuTally-Project?style=social)](https://github.com/aseelalmutari/StuTally-Project/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/aseelalmutari/StuTally-Project)](https://github.com/aseelalmutari/StuTally-Project/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/aseelalmutari/StuTally-Project)](https://github.com/aseelalmutari/StuTally-Project/pulls)

[⬆️ Back to Top](#-stutally)

</div>
