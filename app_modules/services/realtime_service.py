# app/services/realtime_service.py
"""
Real-time analytics service using SocketIO.
Sends periodic updates to connected clients.
"""
from app.extensions import socketio
from database import (
    get_kpis,
    get_students_over_time,
    get_heatmap_data,
    get_stage_comparison_data
)
from apscheduler.schedulers.background import BackgroundScheduler
import logging

logger = logging.getLogger(__name__)

# Background scheduler
scheduler = None


def get_latest_analytics(video_id=None):
    """
    Fetch all analytics data for a given video.
    
    Args:
        video_id: Optional video filter
        
    Returns:
        Dictionary with all analytics data
    """
    try:
        return {
            'status': 'success',
            **get_kpis(video_id),
            'students_over_time': get_students_over_time(video_id),
            'heatmap': get_heatmap_data(video_id),
            'stage_comparison': get_stage_comparison_data()
        }
    except Exception as e:
        logger.error(f"Error fetching analytics: {e}", exc_info=True)
        return {'status': 'error', 'message': str(e)}


def emit_analytics(video_id=None):
    """
    Emit analytics update to all connected clients.
    
    Args:
        video_id: Optional video filter
    """
    try:
        data = get_latest_analytics(video_id)
        socketio.emit('analytics_update', data, namespace='/analytics')
        logger.debug("Analytics update emitted")
    except Exception as e:
        logger.error(f"Error emitting analytics: {e}", exc_info=True)


def init_realtime_service(app):
    """
    Initialize real-time analytics service with background scheduler.
    
    Args:
        app: Flask application instance
        
    Returns:
        BackgroundScheduler instance
    """
    global scheduler
    
    interval = app.config.get('ANALYTICS_UPDATE_INTERVAL', 30)
    
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        emit_analytics,
        'interval',
        seconds=interval,
        id='analytics_update',
        replace_existing=True
    )
    scheduler.start()
    
    logger.info(f"✅ Real-time analytics service started (interval: {interval}s)")
    
    # Register SocketIO events
    register_socketio_events()
    
    return scheduler


def register_socketio_events():
    """Register SocketIO event handlers"""
    
    @socketio.on('connect', namespace='/analytics')
    def handle_connect():
        logger.info("Client connected to analytics namespace")
        # Send initial data on connect
        emit_analytics()
    
    @socketio.on('disconnect', namespace='/analytics')
    def handle_disconnect():
        logger.info("Client disconnected from analytics namespace")
    
    @socketio.on('request_update', namespace='/analytics')
    def handle_request_update(data):
        """Handle manual update requests from client"""
        video_id = data.get('video_id') if data else None
        emit_analytics(video_id)
        logger.debug(f"Manual analytics update requested for video_id={video_id}")


def shutdown_realtime_service():
    """Shutdown background scheduler gracefully"""
    global scheduler
    if scheduler:
        scheduler.shutdown()
        logger.info("Real-time analytics service shut down")

