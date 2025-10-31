# app/routes/main.py
"""
Main routes blueprint.
Handles file upload, video streaming, and homepage.
"""
from flask import Blueprint, render_template, request, jsonify, Response, url_for
import logging
import uuid
import os
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

main_bp = Blueprint('main', __name__)


@main_bp.route('/', methods=['GET', 'POST'])
def index():
    """
    Main page - file upload and processing.
    
    GET: Render upload interface
    POST: Handle file upload (video or image)
    """
    if request.method == 'POST':
        # Import here to avoid circular imports
        from app.services.upload_service import handle_file_upload
        
        try:
            file = request.files.get('file')
            selected_model = request.form.get('model')
            counting_line_enabled = request.form.get('counting_line_enabled') == 'on'
            
            if not file or file.filename == '':
                return jsonify({
                    'status': 'error',
                    'message': 'No file selected'
                }), 400
            
            # Process upload using service
            result = handle_file_upload(
                file=file,
                model_name=selected_model,
                counting_line_enabled=counting_line_enabled
            )
            
            return jsonify(result), result.get('status_code', 200)
            
        except Exception as e:
            logger.error(f"Error in index POST: {e}", exc_info=True)
            return jsonify({
                'status': 'error',
                'message': f'Server error: {str(e)}'
            }), 500
    
    # GET request - render template
    return render_template('index.html')


@main_bp.route('/video_feed/<video_id>')
def video_feed(video_id):
    """
    Stream processed video frames.
    
    Args:
        video_id: Unique video identifier
        
    Returns:
        Streaming response with MJPEG frames
    """
    from app.services.video_service import get_video_stream
    
    try:
        # Get video stream generator
        stream = get_video_stream(video_id)
        
        if stream is None:
            logger.warning(f"No stream available for video_id: {video_id}")
            return Response('', mimetype='multipart/x-mixed-replace; boundary=frame')
        
        return Response(
            stream,
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        logger.error(f"Error in video_feed: {e}", exc_info=True)
        return Response('', mimetype='multipart/x-mixed-replace; boundary=frame')


@main_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded/processed files.
    
    Args:
        filename: Name of file to serve
        
    Returns:
        File response
    """
    from flask import send_from_directory, current_app
    
    try:
        upload_folder = current_app.config['UPLOAD_FOLDER']
        return send_from_directory(upload_folder, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {e}")
        return jsonify({'error': 'File not found'}), 404


@main_bp.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'StuTally',
        'version': '2.0'
    })

