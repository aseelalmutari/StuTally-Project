# app/routes/api.py
"""
API routes blueprint.
JWT-protected API endpoints for external integrations.
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
import logging

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/switch_model', methods=['POST'])
async def switch_model():
    """
    Switch YOLO model asynchronously.
    
    Body:
        {
            "model": "yolov8m.pt"
        }
        
    Returns:
        {
            "status": "success",
            "model": "yolov8m.pt",
            "message": "...",
            "cache_status": {...}
        }
    """
    from ml.model_loader import switch_model_async, get_cache_info
    
    try:
        data = request.get_json()
        model_name = data.get('model')
        
        if not model_name:
            return jsonify({
                'status': 'error',
                'message': 'Model name required'
            }), 400
        
        # Switch model asynchronously
        result = await switch_model_async(model_name)
        
        status_code = 200 if result['status'] == 'success' else 400
        return jsonify(result), status_code
        
    except Exception as e:
        logger.error(f"Error switching model: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/model_status', methods=['GET'])
def model_status():
    """
    Get current model cache status.
    
    Returns:
        {
            "cached_models": [...],
            "cache_size": 2,
            "max_cache_size": 3,
            "device": "cuda",
            "available_models": [...]
        }
    """
    from ml.model_loader import get_cache_info
    
    try:
        cache_info = get_cache_info()
        return jsonify(cache_info)
    except Exception as e:
        logger.error(f"Error getting model status: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/videos', methods=['GET'])
def get_videos():
    """
    Get list of uploaded videos.
    
    Returns:
        [
            {
                "video_id": "...",
                "video_path": "...",
                "datetime": "..."
            },
            ...
        ]
    """
    from database import get_all_videos
    
    try:
        videos = get_all_videos()
        return jsonify(videos)
    except Exception as e:
        logger.error(f"Error fetching videos: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/analytics', methods=['GET'])
@jwt_required()
def api_analytics():
    """
    JWT-protected analytics endpoint.
    
    Headers:
        Authorization: Bearer <token>
    
    Query params:
        video_id: Optional video filter
        
    Returns:
        Analytics data
    """
    from app.services.analytics_service import get_filtered_analytics
    
    try:
        identity = get_jwt_identity()
        logger.info(f"API analytics accessed by: {identity}")
        
        video_id = request.args.get('video_id')
        
        data = get_filtered_analytics(
            video_id=video_id,
            time_range='all_time',
            stat_type='total_students'
        )
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error in API analytics: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@api_bp.route('/clear_cache', methods=['POST'])
def clear_model_cache():
    """
    Clear model cache and free memory.
    Use with caution in production.
    
    Returns:
        {
            "status": "success",
            "message": "..."
        }
    """
    from ml.model_loader import clear_model_cache
    
    try:
        result = clear_model_cache()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error clearing cache: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

