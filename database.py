# database.py

import sqlite3
import os
import logging

# تحديد مسار قاعدة البيانات
DATA_FOLDER = 'data'
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
DB_PATH = os.path.join(DATA_FOLDER, 'detections.db')

# إعدادات السجلات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # إنشاء جدول 'detections' مع العمود الجديد 'media_type'
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                class_name TEXT,
                confidence REAL,
                x1 INTEGER,
                y1 INTEGER,
                x2 INTEGER,
                y2 INTEGER,
                track_id INTEGER,
                stage TEXT,
                datetime TEXT DEFAULT CURRENT_TIMESTAMP,
                action TEXT,
                media_type TEXT  -- العمود الجديد
            )
        ''')
        # إنشاء جدول 'videos' لتخزين معلومات الفيديوهات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                video_path TEXT,
                counting_line_enabled BOOLEAN,
                datetime TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # إنشاء جدول 'daily_statistics' لتخزين الإحصائيات اليومية
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                class_name TEXT,
                stage TEXT,
                count INTEGER,
                entries INTEGER,
                exits INTEGER
            )
        ''')
        conn.commit()
        logger.info("تم تهيئة قاعدة البيانات بنجاح مع تعديل عمود 'media_type' في جدول 'detections'.")

def log_detection(video_id, class_name, confidence, x1, y1, x2, y2, track_id, stage, action):
    try:
        # تحديد نوع الوسائط بناءً على video_id
        media_type = 'image' if video_id.startswith('image_upload_') else 'video'
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections (video_id, class_name, confidence, x1, y1, x2, y2, track_id, stage, action, media_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (video_id, class_name, confidence, x1, y1, x2, y2, track_id, stage, action, media_type))
            conn.commit()
            logger.info(f"تم تسجيل الكشف: Video ID={video_id}, Track ID={track_id}, Class={class_name}, Stage={stage}, Action={action}, Media Type={media_type}")
    except Exception as e:
        logger.error(f"Error logging detection: {e}")

def save_video_info(video_id, video_path, counting_line_enabled):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO videos (video_id, video_path, counting_line_enabled)
                VALUES (?, ?, ?)
            ''', (video_id, video_path, counting_line_enabled))
            conn.commit()
            logger.info(f"تم حفظ معلومات الفيديو: Video ID={video_id}, Path={video_path}, Counting Line Enabled={counting_line_enabled}")
    except Exception as e:
        logger.error(f"Error saving video info: {e}")

def get_latest_video_id():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT video_id FROM videos ORDER BY datetime DESC LIMIT 1
            ''')
            result = cursor.fetchone()
            if result:
                logger.info(f"تم استرجاع أحدث video_id: {result[0]}")
                return result[0]
            else:
                logger.info("No video_id found in the database.")
                return None
    except Exception as e:
        logger.error(f"Error fetching latest video_id: {e}")
        return None

def get_all_videos():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT video_id, video_path, datetime FROM videos ORDER BY datetime DESC
            ''')
            results = cursor.fetchall()
            videos = [{'video_id': row[0], 'video_path': row[1], 'datetime': row[2]} for row in results]
            logger.info(f"تم جلب {len(videos)} فيديوهات من قاعدة البيانات.")
            return videos
    except Exception as e:
        logger.error(f"Error fetching videos: {e}")
        return []

def get_analytics(video_id=None):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            query = '''
                SELECT class_name, stage, COUNT(*) as count
                FROM detections
            '''
            params = []
            conditions = []

            if video_id:
                conditions.append('video_id = ?')
                params.append(video_id)

            if conditions:
                query += ' WHERE ' + ' AND '.join(conditions)

            query += ' GROUP BY class_name, stage'

            cursor.execute(query, params)
            data = cursor.fetchall()
            logger.info(f"تم استرجاع التحليلات: {data}")
            return data
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}")
        return []

def get_time_based_analytics(video_id=None, time_range='daily'):
    # يمكنك تنفيذ هذه الدالة بناءً على احتياجاتك
    pass

def get_date_based_analytics(video_id=None, start_date=None, end_date=None):
    # يمكنك تنفيذ هذه الدالة بناءً على احتياجاتك
    pass

def log_daily_statistics(date, class_name, stage, count, entries, exits):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO daily_statistics (date, class_name, stage, count, entries, exits)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (date, class_name, stage, count, entries, exits))
            conn.commit()
            logger.info(f"تم تسجيل الإحصائيات اليومية: Date={date}, Class={class_name}, Stage={stage}, Count={count}")
    except Exception as e:
        logger.error(f"Error logging daily statistics: {e}")

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

            return {
                'total_students': total_students,
                'total_entries': total_entries,
                'total_exits': total_exits
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
            import numpy as np
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
