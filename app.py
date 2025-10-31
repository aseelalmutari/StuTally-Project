# app.py

# Real-time and authentication modules
from auth import auth_bp, login_manager, bcrypt, jwt

# Flask and login utilities
from flask import (
    Flask,
    render_template,
    Response,
    request,
    jsonify,
    send_file,
    url_for,
    flash
)
from flask_login import login_required

# Computer vision and model utilities
import cv2
import os

# تعطيل التحديثات التلقائية لـ YOLO
os.environ['YOLO_VERBOSE'] = 'False'

from ultralytics import YOLO
from database import (
    init_db,
    log_detection,
    get_analytics,
    get_time_based_analytics,
    get_date_based_analytics,
    log_daily_statistics,
    save_video_info,
    get_latest_video_id,
    get_all_videos
)
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import numpy as np
import threading
import logging
import uuid
from datetime import datetime, date
from apscheduler.schedulers.background import BackgroundScheduler
import sqlite3

# Custom model inference
from inference import get_model
import supervision as sv

# Flask app initialization
app = Flask(__name__)

# SocketIO for live updates
from flask_socketio import SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Authentication setup
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
bcrypt.init_app(app)
app.config['JWT_SECRET_KEY'] = 'super-secret-key'
jwt.init_app(app)
app.register_blueprint(auth_bp)
app.secret_key = 'your_secret_key'

# Folders and database path
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
DATA_FOLDER = 'data'
DB_PATH = os.path.join(DATA_FOLDER, 'detections.db')

for folder in [UPLOAD_FOLDER, MODELS_FOLDER, DATA_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODELS_FOLDER'] = MODELS_FOLDER

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

# النماذج المتاحة من YOLO
YOLO_MODELS = {
    'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
    'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
    'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
    'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt',
    'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt',
    # أضف أي نماذج مخصصة أخرى إذا كان لديك
}

# تعريف خريطة أسماء الفئات للنموذج المخصص
custom_model_class_names = {
    "High": "High School",
    "Middle": "Middle School",
    "students": "Unknown"  # أو أي تسمية أخرى تناسبك
}

# المتغيرات العالمية
current_model_name = 'yolov8s.pt'  # النموذج الافتراضي
current_yolo = None
model_lock = threading.Lock()
current_video_id = None
current_video_path = None
tracker_lock = threading.Lock()

# متغيرات العد
line_enabled = False
line_position = None
previous_positions = {}
entry_counts = {}
exit_counts = {}
total_counts = {}

# تهيئة متتبع DeepSort
tracker = DeepSort(max_age=15, n_init=3, max_iou_distance=0.7)
tracked_ids = set()

# خريطة من track_id إلى label و stage
track_labels = {}

# جدولة الإحصائيات اليومية
scheduler = BackgroundScheduler()

def scheduled_daily_statistics():
    """
    وظيفة المجدول لتسجيل الإحصائيات اليومية عند منتصف الليل.
    """
    today = date.today().isoformat()
    #logger.info(f"Logging daily statistics for date: {today}")
    analytics = get_analytics()
    for item in analytics:
        class_name, stage, count = item
        # نفترض أن الإحصائيات اليومية تشمل فقط العدد الإجمالي
        log_daily_statistics(today, class_name, stage, count, 0, 0)

scheduler.add_job(scheduled_daily_statistics, 'cron', hour=0, minute=0)
scheduler.start()
#logger.info("Scheduler started for daily statistics.")

def ensure_model_exists(model_name):
    """
    التأكد من أن النموذج المطلوب موجود في مجلد النماذج.
    """
    model_path = os.path.join(MODELS_FOLDER, model_name)
    if not os.path.exists(model_path):
        if model_name in YOLO_MODELS and YOLO_MODELS[model_name].startswith('http'):
            download_model(model_name, YOLO_MODELS[model_name], model_path)
        else:
            logger.error(f"Model '{model_name}' not found in '{MODELS_FOLDER}/'. Please ensure it is present.")
            return False
    return True

def download_model(model_name, url, save_path):
    """
    وظيفة لتنزيل النموذج من URL وحفظه في المسار المحدد.
    """
    try:
        import requests
        logger.info(f"Downloading {model_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"Downloaded {model_name} successfully.")
    except Exception as e:
        logger.error(f"Failed to download model '{model_name}': {e}")
        raise

def load_yolo_model(model_name):
    """
    وظيفة لتحميل نموذج YOLO من المسار المحدد.
    """
    import time
    
    global current_yolo
    try:
        model_path = os.path.join(MODELS_FOLDER, model_name)
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        logger.info(f"Model file size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        logger.info(f"Starting to load model from: {model_path}")
        yolo_model = YOLO(model_path)
        logger.info(f"Model object created, moving to device: {device}")
        yolo_model.to(device)
        logger.info(f"YOLO model '{model_name}' loaded successfully.")
        logger.info(f"Model classes: {yolo_model.names}")
        return yolo_model
            
    except Exception as e:
        logger.error(f"Error loading YOLO model '{model_name}': {e}", exc_info=True)
        return None

# تحميل النموذج الافتراضي
if ensure_model_exists(current_model_name):
    current_yolo = load_yolo_model(current_model_name)
else:
    logger.error(f"Cannot load model '{current_model_name}'. Exiting.")
    exit(1)

def check_line_crossing(y_previous, y_current, line_y):
    """
    وظيفة للتحقق مما إذا كان كائن ما قد عبر خط العد.
    """
    if y_previous < line_y and y_current >= line_y:
        return "down"
    elif y_previous > line_y and y_current <= line_y:
        return "up"
    return None

def gen_frames(video_path, video_id):
    """
    وظيفة لتوليد إطارات الفيديو ومعالجتها باستخدام YOLO و DeepSort.
    """
    global current_video_id, line_enabled, line_position, previous_positions
    global entry_counts, exit_counts, total_counts, tracker, tracked_ids, current_yolo, track_labels
    cap = cv2.VideoCapture(video_path)
    frame_skip = 3
    frame_count = 0

    if not cap.isOpened():
        logger.error(f"Cannot open video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_position = (0, frame_height // 2, frame_width, frame_height // 2)  # خط العد في المنتصف

    while True:
        # التحقق مما إذا كان video_id قد تغير
        if video_id != current_video_id:
            logger.info(f"Stopping video feed for video_id: {video_id}")
            break

        success, frame = cap.read()
        if not success:
            logger.info(f"Reached end of video: {video_path}. Stopping.")
            break  # الخروج من الحلقة بدلاً من إعادة التشغيل

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with model_lock:
            yolo = current_yolo

        if yolo is not None:
            try:
                results = yolo.predict(source=frame_rgb, device=device)
                detections = results[0].boxes if results and len(results) > 0 else []
            except Exception as e:
                logger.error(f"Error during YOLO prediction: {e}")
                detections = []

            detections_list = []
            for box in detections:
                try:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    label = yolo.names.get(class_id, "Unknown")  # التأكد من أن label هو اسم الفئة

                    # تضمين فقط الفئات المدعومة
                    if current_model_name == 'best.pt':
                        if label in custom_model_class_names.keys() or label == "students":
                            stage = custom_model_class_names.get(label, "Unknown")
                            detections_list.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id, label, stage))
                    else:
                        if label == "person":  # تعديل بناءً على فئات YOLO القياسية
                            detections_list.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id, label, None))
                except Exception as e:
                    logger.error(f"Error processing detection box: {e}")

            with tracker_lock:
                try:
                    tracks = tracker.update_tracks(detections_list, frame=frame)
                except Exception as e:
                    logger.error(f"Error updating tracks: {e}")
                    tracks = []

                for track in tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue

                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, ltrb)
                    y_center = (y1 + y2) // 2
                    track_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # استرجاع label و stage من track_labels أو تعيين الافتراضي
                    label = track_labels.get(track_id, {}).get('label', 'Unknown')
                    stage = track_labels.get(track_id, {}).get('stage', None)

                    # تعيين label و stage إذا لم يتم تعيينهما بعد
                    if label == 'Unknown' and current_model_name == 'best.pt':
                        # محاولة المطابقة مع الكشف
                        matched = False
                        for det in detections_list:
                            bbox, conf, cls_id, lbl, stg = det
                            det_center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
                            distance = np.linalg.norm(np.array(track_center) - np.array(det_center))
                            if distance < 50:  # عتبة المسافة
                                label = lbl
                                stage = stg
                                matched = True
                                break
                        if not matched:
                            label = 'Unknown'
                            stage = None

                        # تعيين إلى track_labels
                        track_labels[track_id] = {
                            'label': label,
                            'stage': stage
                        }
                        logger.debug(f"Track ID {track_id} assigned label '{label}' and stage '{stage}'.")

                    elif label == 'Unknown' and current_model_name != 'best.pt':
                        # بالنسبة لنماذج YOLO القياسية، label هو 'person' و stage هو None
                        label = 'person'
                        stage = None
                        track_labels[track_id] = {
                            'label': label,
                            'stage': stage
                        }

                    # تحديث العدادات بناءً على عبور الخط
                    if label not in entry_counts:
                        entry_counts[label] = 0
                    if label not in exit_counts:
                        exit_counts[label] = 0
                    if label not in total_counts:
                        total_counts[label] = 0

                    if line_enabled:
                        line_x1, line_y1, line_x2, line_y2 = line_position
                        if track_id in previous_positions:
                            y_previous = previous_positions[track_id]
                            direction = check_line_crossing(y_previous, y_center, line_y1)
                            if direction == "down" and track_id not in tracked_ids:
                                tracked_ids.add(track_id)
                                exit_counts[label] += 1
                                log_detection(video_id, label, confidence, x1, y1, x2, y2, track_id, stage, action='exit')
                                color = (0, 0, 255)  # أحمر
                                logger.info(f"Track ID {track_id} ({label}) exited. Total exits: {exit_counts[label]}")
                            elif direction == "up" and track_id not in tracked_ids:
                                tracked_ids.add(track_id)
                                entry_counts[label] += 1
                                log_detection(video_id, label, confidence, x1, y1, x2, y2, track_id, stage, action='entry')
                                color = (0, 255, 0)  # أخضر
                                logger.info(f"Track ID {track_id} ({label}) entered. Total entries: {entry_counts[label]}")
                            else:
                                color = (255, 255, 255)  # أبيض
                        else:
                            color = (255, 255, 255)  # أبيض

                        previous_positions[track_id] = y_center

                        # رسم خط العد
                        cv2.line(frame, (line_x1, line_y1), (line_x2, line_y2), (0, 0, 255), 2)
                    else:
                        color = (255, 255, 255)  # أبيض
                        if track_id not in tracked_ids:
                            tracked_ids.add(track_id)
                            total_counts[label] += 1
                            log_detection(video_id, label, confidence, x1, y1, x2, y2, track_id, stage, action='counted')
                            logger.info(f"Track ID {track_id} ({label}) counted. Total count: {total_counts[label]}")

                    # رسم المستطيل والتسمية
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {track_id}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # عرض العدادات على الفيديو
                if line_enabled:
                    y_offset = 30
                    for label in entry_counts:
                        cv2.putText(frame, f'Entries ({label}): {entry_counts[label]}', (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        y_offset += 30
                    for label in exit_counts:
                        cv2.putText(frame, f'Exits ({label}): {exit_counts[label]}', (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        y_offset += 30
                else:
                    y_offset = 30
                    for label in total_counts:
                        cv2.putText(frame, f'Total ({label}): {total_counts[label]}', (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        y_offset += 30

        # تشفير الإطار إلى JPEG
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            logger.error("Failed to encode frame to JPEG.")
            continue
        frame = buffer.tobytes()
        try:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except GeneratorExit:
            logger.info(f"GeneratorExit: Stopping frame generation for video_id: {video_id}")
            break
        except Exception as e:
            logger.error(f"Exception during frame streaming: {e}")
            break

    cap.release()
    logger.info(f"Video capture released for video_id: {video_id}")

@app.route('/video_feed/<video_id>')
def video_feed(video_id):
    """
    مسار لبث الفيديو مباشرة.
    """
    global current_video_path, current_video_id
    if not current_video_path or not os.path.exists(current_video_path):
        logger.info("No video uploaded yet. Sending empty response.")
        return Response('', mimetype='multipart/x-mixed-replace; boundary=frame')
    if video_id != current_video_id:
        logger.info("Video ID mismatch. Sending empty response.")
        return Response('', mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen_frames(current_video_path, video_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    الصفحة الرئيسية للتعامل مع تحميل الفيديوهات والصور.
    """
    global current_model_name, current_yolo, tracked_ids
    global entry_counts, exit_counts, total_counts
    global current_video_id, line_enabled, current_video_path, tracker, track_labels

    if request.method == 'POST':
        file = request.files.get('file')
        selected_model = request.form.get('model')
        counting_line_enabled = request.form.get('counting_line_enabled') == 'on'

        if not file:
            logger.error("No file part in the request.")
            flash('No file selected.', 'danger')
            return render_template('index.html')

        if file.filename == '':
            logger.error("No selected file.")
            flash('No file selected.', 'danger')
            return render_template('index.html')

        if selected_model and selected_model != current_model_name:
            logger.info(f"🔄 Switching model from '{current_model_name}' to '{selected_model}'")
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            
            success = update_model(selected_model)
            
            if not success:
                logger.error(f"❌ Failed to update model to: {selected_model}")
                return jsonify({'status': 'error', 'message': f'Failed to load model: {selected_model}. Please try again or select a different model.'}), 500
            
            logger.info(f"✅ Successfully switched to model: {selected_model}")
            sys.stdout.flush()
            sys.stderr.flush()

        if file:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                # معالجة تحميل الفيديو
                video_id = str(uuid.uuid4())
                current_video_id = video_id
                video_filename = f'video_{video_id}{file_ext}'
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
                current_video_path = video_path
                logger.info(f"Attempting to save video to path: {video_path}")
                try:
                    file.save(video_path)
                    logger.info(f"Video saved to {video_path} with video_id: {video_id}.")

                    line_enabled = counting_line_enabled  # تحديث الحالة العالمية

                    # حفظ معلومات الفيديو في قاعدة البيانات
                    save_video_info(video_id, video_path, counting_line_enabled)

                    # إعادة تهيئة متتبع DeepSort
                    with tracker_lock:
                        logger.info("Reinitializing DeepSort tracker for new video.")
                        tracker = DeepSort(max_age=15, n_init=3, max_iou_distance=0.7)
                        tracked_ids = set()
                        entry_counts = {}
                        exit_counts = {}
                        total_counts = {}
                        previous_positions = {}
                        track_labels = {}

                    return jsonify({
                        'status': 'success',
                        'message': 'Video uploaded successfully!',
                        'video_id': video_id  # تضمين video_id في الاستجابة
                    }), 200
                except Exception as e:
                    logger.error(f"Error saving video: {e}", exc_info=True)
                    return jsonify({'status': 'error', 'message': f'Failed to save video: {str(e)}'}), 500

            elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                # معالجة تحميل الصور
                image_id = str(uuid.uuid4())
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{image_id}{file_ext}')
                try:
                    # تعيين current_video_id إلى None لإيقاف أي بث فيديو جارٍ
                    logger.info(f"Stopping any active video feed before processing image...")
                    old_video_id = current_video_id
                    current_video_id = None
                    if old_video_id:
                        logger.info(f"Stopped video feed: {old_video_id}")
                    import time
                    time.sleep(0.5)  # انتظار قصير للسماح للفيديو بالإيقاف

                    file.save(image_path)
                    logger.info(f"Image saved to {image_path} with image_id: {image_id}.")

                    # إنشاء video_id فريد للصورة
                    video_id = 'image_upload_' + image_id

                    if selected_model == 'best.pt':
                        # معالجة الصورة باستخدام النموذج المخصص والحصول على class_type
                        annotated_image_path, class_type = process_image_with_custom_model(image_path, video_id)
                        # نفترض أن class_type يتوافق مع stage
                        stage = class_type.replace(' & Middle School', '')  # تبسيط stage
                    else:
                        # معالجة الصورة باستخدام YOLO القياسي
                        annotated_image_path = process_image_with_yolo(image_path, video_id)
                        class_type = None  # لا يوجد class_type للنماذج الأخرى
                        stage = None

                    # حفظ معلومات تحميل الصورة في جدول الفيديوهات
                    save_video_info(video_id=video_id, video_path=image_path, counting_line_enabled=False)

                    response_data = {
                        'status': 'success',
                        'message': 'Image processed successfully!',
                        'processed_image': url_for('uploaded_file', filename=os.path.basename(annotated_image_path))
                    }

                    if selected_model == 'best.pt':
                        response_data['class_type'] = class_type  # تضمين class_type في الاستجابة

                    return jsonify(response_data), 200
                except Exception as e:
                    logger.error(f"Error processing image: {e}", exc_info=True)
                    return jsonify({'status': 'error', 'message': f'Failed to process image: {str(e)}'}), 500
            else:
                #logger.error("Unsupported file type.")
                return jsonify({'status': 'error', 'message': 'Unsupported file type.'}), 400

    return render_template('index.html')

def process_image_with_custom_model(image_path, video_id):
    """
    معالجة الصورة باستخدام النموذج المخصص وإرجاع مسار الصورة المشروحة ونوع الفئة ("High School" أو "Middle School").
    """
    api_key = "Y4FNUG8Esbj83F65kkBA"  # استبدلها بمفتاح Roboflow API الخاص بك
    model_id = "ss-uniform/3"  # استبدلها بمعرف النموذج الخاص بك

    logger.info("Starting custom model processing...")
    try:
        # قراءة الصورة
        image = cv2.imread(image_path)

        # التحقق من تحميل الصورة بنجاح
        if image is None:
            raise FileNotFoundError(f"Image file '{image_path}' not found or could not be opened.")

        # تحميل النموذج المدرب مسبقًا باستخدام مفتاح API
        model = get_model(model_id=model_id, api_key=api_key)

        # إجراء الاستدلال على الصورة
        results = model.infer(image)[0]

        # تحميل النتائج إلى واجهة Supervision Detections API
        detections = sv.Detections.from_inference(results)

        # إنشاء مُعالج Supervision
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # شرح الصورة بنتائج الاستدلال
        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # **إنشاء خريطة من class_id إلى class_name**
        id_to_class_name = {
            0: "High",
            1: "Middle",
            2: "students"
        }

        # تحديد نوع الفئة بناءً على الكشف
        detected_classes = detections.class_id  # نفترض أن class_id هو رقم معرف للفئة
        class_names = [id_to_class_name.get(class_id, "Unknown") for class_id in detected_classes]

        #logger.info(f"class_names: {class_names}")

        # تحديد نوع الفئة المدمج
        unique_class_names = set(class_names)
        if "High" in unique_class_names and "Middle" in unique_class_names:
            class_type = "High School & Middle School"
        elif "High" in unique_class_names:
            class_type = "High School"
        elif "Middle" in unique_class_names:
            class_type = "Middle School"
        else:
            class_type = "Unknown"

        #logger.info(f"class_type: {class_type}")

        # إضافة تعليق على الصورة
        cv2.putText(
            annotated_image,
            f"Educational Level: {class_type}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # حفظ الصورة المشروحة
        annotated_image_path = os.path.splitext(image_path)[0] + '_processed.jpg'
        cv2.imwrite(annotated_image_path, annotated_image)
        logger.info("Custom model processing completed successfully.")

        # تسجيل الكشف مع stage
        for class_id, confidence, box in zip(detections.class_id, detections.confidence, detections.xyxy):
            class_name = id_to_class_name.get(class_id, "Unknown")
            stage = custom_model_class_names.get(class_name, "Unknown") if class_name != "students" else "Unknown"
            x1, y1, x2, y2 = map(int, box)
            # استخدام track_id وهمي لأننا نتعامل مع صورة
            track_id = 0
            #logger.info(f"Logging detection: class_id={class_id}, class_name={class_name}, stage={stage}")
            log_detection(video_id, class_name, confidence, x1, y1, x2, y2, track_id, stage, action='counted')

        return annotated_image_path, class_type  # إرجاع مسار الصورة المشروحة ونوع الفئة
    except Exception as e:
        logger.error(f"Error in custom model processing: {e}", exc_info=True)
        raise

def process_image_with_yolo(image_path, video_id):
    """
    معالجة الصورة باستخدام نموذج YOLO القياسي وإرجاع مسار الصورة المشروحة.
    """
    with model_lock:
        yolo = current_yolo
        if yolo is None:
            logger.error("YOLO model is not loaded.")
            raise ValueError('YOLO model is not loaded.')

    try:
        results = yolo.predict(source=image_path, device=device, conf=0.5)
        if not results or len(results) == 0:
            logger.error("No results from YOLO prediction.")
            raise ValueError('No detections found.')

        annotated_frame = results[0].plot()

        # تحديد مرحلة
        stage = "Unknown"  # لأن النموذج القياسي لا يحدد مرحلة

        # إضافة تعليق على الصورة
        cv2.putText(
            annotated_frame,
            f"Educational Level: {stage}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # حفظ الصورة المشروحة
        annotated_image_path = os.path.splitext(image_path)[0] + '_processed.jpg'
        cv2.imwrite(annotated_image_path, annotated_frame)
        logger.info("YOLO image processing completed successfully.")

        # تسجيل الكشف مع stage كـ None لأن النموذج القياسي لا يوفر هذه المعلومات
        detections = results[0].boxes
        for box in detections:
            class_id = int(box.cls[0])
            class_name = yolo.names.get(class_id, "Unknown")
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            log_detection(video_id, class_name, confidence, x1, y1, x2, y2, 0, None, action='counted')

        return annotated_image_path
    except Exception as e:
        logger.error(f"Error during YOLO image processing: {e}")
        raise e

def update_model(model):
    """
    تحديث النموذج الحالي إلى النموذج المحدد.
    """
    import time
    start_time = time.time()
    
    global current_model_name, current_yolo, tracker, tracked_ids, entry_counts, exit_counts, total_counts, previous_positions, track_labels
    if model not in YOLO_MODELS and model != 'best.pt':
        logger.error(f"Invalid model selected: {model}")
        return False

    with model_lock:
        if model == current_model_name:
            logger.info(f"Model '{model}' is already loaded.")
            return True

        logger.info(f"⏳ Starting model switch from '{current_model_name}' to '{model}'...")
        
        # تحرير النموذج القديم من الذاكرة
        old_yolo = current_yolo
        if old_yolo is not None:
            try:
                del old_yolo
                logger.info("Old model removed from memory")
            except Exception as e:
                logger.warning(f"Could not delete old model: {e}")
        
        # تنظيف ذاكرة GPU إذا كانت متاحة
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

        if model == 'best.pt':
            # 'best.pt' هو نموذج مخصص يعالج بشكل مختلف
            current_model_name = model
            current_yolo = None  # تعيينه إلى None لأننا نستخدم نموذج مخصص
            logger.info("Switched to custom model 'best.pt'")
        else:
            new_model_path = os.path.join(MODELS_FOLDER, model)
            if not os.path.exists(new_model_path):
                logger.info(f"Model '{model}' not found locally. Attempting to download.")
                url = YOLO_MODELS.get(model)
                if not url or not url.startswith('http'):
                    logger.error(f"No download URL found for model: {model}")
                    return False
                try:
                    download_model(model, url, new_model_path)
                except Exception as e:
                    logger.error(f"Failed to download model '{model}': {e}")
                    return False

            try:
                logger.info(f"Loading model '{model}'... (this may take 10-60 seconds)")
                new_yolo = load_yolo_model(model)
                if new_yolo is None:
                    logger.error(f"Failed to load model '{model}'. Model returned None.")
                    logger.error("Possible causes: model file is corrupted, missing dependencies, or YOLO version mismatch")
                    return False
                logger.info(f"Model '{model}' loaded successfully with {len(new_yolo.names)} classes")
            except Exception as e:
                logger.error(f"Failed to load the selected model: {model} - {e}", exc_info=True)
                logger.error("Try restarting the application or using a different model")
                return False

            current_model_name = model
            current_yolo = new_yolo

        # إعادة تهيئة متتبع DeepSort بعد تحديث النموذج
        with tracker_lock:
            logger.info("Reinitializing DeepSort tracker after model update.")
            tracker = DeepSort(max_age=15, n_init=3, max_iou_distance=0.7)
            tracked_ids = set()
            entry_counts = {}
            exit_counts = {}
            total_counts = {}
            previous_positions = {}
            track_labels = {}

    elapsed_time = time.time() - start_time
    logger.info(f"✅ Model successfully updated to '{model}' in {elapsed_time:.2f} seconds")
    return True

@app.route('/get_classes')
def get_classes():
    """
    مسار للحصول على قائمة الفئات المدعومة من النموذج الحالي.
    """
    if current_yolo is not None:
        classes = current_yolo.names
        return jsonify({'status': 'success', 'classes': classes})
    else:
        return jsonify({'status': 'error', 'message': 'No model loaded.'}), 500

# Protected analytics routes
@app.route('/analytics')
@app.route('/analytics/')
@login_required
def analytics():
    return render_template('analytics.html')

@app.route('/analytics/data')
@login_required
def analytics_data():
    """
    مسار لتوفير بيانات التحليلات بصيغة JSON.
    """
    video_id = request.args.get('video_id')
    time_range = request.args.get('time_range', 'all_time')
    date_str = request.args.get('date')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    stat_type = request.args.get('stat_type', 'total_students')

    if stat_type == 'academic_stages' and current_model_name == 'best.pt':
        # استرجاع بيانات المراحل التعليمية فقط إذا كان النموذج المستخدم هو 'best.pt'
        data = get_analytics(video_id=video_id)
    elif stat_type == 'academic_stages' and current_model_name != 'best.pt':
        # إذا كان stat_type هو 'academic_stages' ولكن النموذج ليس 'best.pt'
        return jsonify({'status': 'error', 'message': 'Academic stages are only available with the best.pt model.'}), 400
    else:
        if time_range == 'all_time':
            data = get_analytics(video_id=video_id)
        elif time_range in ['daily', 'weekly', 'monthly', 'morning', 'afternoon']:
            data = get_time_based_analytics(video_id=video_id, time_range=time_range)
        elif time_range == 'date_based':
            # نفترض أن اختيار التاريخ يشمل start_date و end_date
            data = get_date_based_analytics(video_id=video_id, start_date=date_str, end_date=date_str)
        else:
            data = get_analytics(video_id=video_id)

    return jsonify(data)

@app.route('/analytics/kpis')
@login_required
def analytics_kpis():
    video_id = request.args.get('video_id')
    data = get_kpis(video_id=video_id)
    return jsonify(data)

@app.route('/analytics/students_over_time')
@login_required
def analytics_students_over_time():
    video_id = request.args.get('video_id')
    data = get_students_over_time(video_id=video_id)
    return jsonify(data)

@app.route('/analytics/heatmap_data')
@login_required
def analytics_heatmap_data():
    video_id = request.args.get('video_id')
    data = get_heatmap_data(video_id=video_id)
    return jsonify(data)

@app.route('/analytics/stage_comparison_data')
@login_required
def analytics_stage_comparison_data():
    # تعديل المسار ليتوافق مع الدالة المعدلة
    data = get_stage_comparison_data()
    return jsonify(data)

def get_kpis(video_id=None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()

            # إجمالي عدد الطلاب
            query_total_students = '''
                SELECT COUNT(DISTINCT track_id) FROM detections
            '''
            # إجمالي الدخول
            query_total_entries = '''
                SELECT COUNT(*) FROM detections WHERE action = 'entry'
            '''
            # إجمالي الخروج
            query_total_exits = '''
                SELECT COUNT(*) FROM detections WHERE action = 'exit'
            '''

            params = []
            conditions = []

            if video_id:
                conditions.append('video_id = ?')
                params.append(video_id)
                query_total_students += ' WHERE ' + ' AND '.join(conditions)
                query_total_entries += ' AND ' + ' AND '.join(conditions)
                query_total_exits += ' AND ' + ' AND '.join(conditions)

            # تنفيذ الاستعلامات
            cursor.execute(query_total_students, params)
            total_students = cursor.fetchone()[0]

            cursor.execute(query_total_entries, params)
            total_entries = cursor.fetchone()[0]

            cursor.execute(query_total_exits, params)
            total_exits = cursor.fetchone()[0]

            # يمكنك إضافة استعلامات أخرى لنسب المراحل التعليمية

            return {
                'total_students': total_students,
                'total_entries': total_entries,
                'total_exits': total_exits
                # أضف البيانات الأخرى إذا لزم الأمر
            }
    except Exception as e:
        logger.error(f"Error fetching KPIs: {e}")
        return {}

def get_students_over_time(video_id=None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT datetime, COUNT(DISTINCT track_id) as count
                FROM detections
            '''
            params = []
            conditions = []

            if video_id:
                conditions.append('video_id = ?')
                params.append(video_id)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            query += ' GROUP BY datetime'

            cursor.execute(query, params)
            data = cursor.fetchall()

            timestamps = [row[0] for row in data]
            counts = [row[1] for row in data]

            return {'timestamps': timestamps, 'counts': counts}
    except Exception as e:
        logger.error(f"Error fetching students over time data: {e}")
        return {}

def get_heatmap_data(video_id=None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT strftime('%w', datetime) as day_of_week,
                       strftime('%H', datetime) as hour_of_day,
                       COUNT(*) as count
                FROM detections
            '''
            params = []
            conditions = []

            if video_id:
                conditions.append('video_id = ?')
                params.append(video_id)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            query += ' GROUP BY day_of_week, hour_of_day'

            cursor.execute(query, params)
            data = cursor.fetchall()

            days = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            hours = [str(i) for i in range(24)]
            counts = np.zeros((7, 24))

            for row in data:
                day = int(row[0])
                hour = int(row[1])
                count = row[2]
                counts[day, hour] = count

            return {'days': days, 'hours': hours, 'counts': counts.tolist()}
    except Exception as e:
        logger.error(f"Error fetching heatmap data: {e}")
        return {}

def get_stage_comparison_data():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT stage, COUNT(*) as count
                FROM detections
                WHERE media_type = 'image'
                GROUP BY stage
            '''
            cursor.execute(query)
            data = cursor.fetchall()

            stages = [row[0] if row[0] else 'Unknown' for row in data]
            counts = [row[1] for row in data]

            return {'stages': stages, 'counts': counts}
    except Exception as e:
        logger.error(f"Error fetching stage comparison data: {e}")
        return {}
# Videos list
@app.route('/videos')
@login_required
def get_videos():
    """
    مسار للحصول على قائمة الفيديوهات المرفوعة (بما في ذلك تحميل الصور).
    """
    videos = get_all_videos()
    return jsonify(videos)

@app.route('/analytics/download')
@login_required
def analytics_download():
    """
    مسار لتنزيل بيانات التحليلات كملف CSV.
    """
    video_id = request.args.get('video_id')
    stat_type = request.args.get('stat_type', 'total_students')
    data = get_analytics(video_id=video_id)  # يمكن تعديل الاستعلام بناءً على stat_type إذا لزم الأمر
    df = pd.DataFrame(data, columns=['class_name', 'stage', 'count'])
    csv_data = df.to_csv(index=False)
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=analytics.csv"}
    )

@app.route('/analytics/download_pdf')
@login_required
def analytics_download_pdf():
    """
    مسار لتنزيل بيانات التحليلات كملف PDF.
    """
    video_id = request.args.get('video_id')
    stat_type = request.args.get('stat_type', 'total_students')
    data = get_analytics(video_id=video_id)  # يمكن تعديل الاستعلام بناءً على stat_type إذا لزم الأمر
    df = pd.DataFrame(data, columns=['class_name', 'stage', 'count'])
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    for index, row in df.iterrows():
        text = f"Category: {row['class_name']} - Stage: {row['stage']} - Count: {row['count']}"
        c.drawString(50, y, text)
        y -= 20
        if y < 50:
            c.showPage()
            y = height - 50

    c.save()
    buffer.seek(0)
    return send_file(
        buffer,
        as_attachment=True,
        download_name='analytics.pdf',
        mimetype='application/pdf'
    )
# Static uploads
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    مسار لخدمة الملفات المرفوعة والمشروحة.
    """
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

# App entry point
if __name__ == '__main__':
    init_db()
    current_video_id = get_latest_video_id()
    if current_video_id:
        # تعيين current_video_path بناءً على video_id
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT video_path FROM videos WHERE video_id = ?', (current_video_id,))
            result = cursor.fetchone()
            if result:
                current_video_path = result[0]
                logger.info(f"Set current_video_id to latest video: {current_video_id}")
                logger.info(f"Set current_video_path to: {current_video_path}")
            else:
                logger.info("No video_path found for the latest video_id.")
    else:
        logger.info("No existing video_id found.")
    
    # استيراد realtime بعد تهيئة socketio لتجنب circular import
    try:
        import realtime
        logger.info("Realtime module loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load realtime module: {e}")
    
    # تشغيل مع threading لمنع blocking
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)