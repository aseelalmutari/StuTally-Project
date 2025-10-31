# realtime.py
from database import (
    get_kpis,
    get_students_over_time,
    get_heatmap_data,
    get_stage_comparison_data
)
from app import socketio
from apscheduler.schedulers.background import BackgroundScheduler

def get_latest_analytics(video_id=None):
    return {
        **get_kpis(video_id),
        'students_over_time': get_students_over_time(video_id),
        'heatmap': get_heatmap_data(video_id),
        'stage_comparison': get_stage_comparison_data()
    }

def emit_analytics():
    data = get_latest_analytics()
    socketio.emit('analytics_update', data)

scheduler = BackgroundScheduler()
scheduler.add_job(emit_analytics, 'interval', seconds=30)
scheduler.start()
