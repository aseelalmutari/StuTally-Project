# app/services/upload_service.py
"""
File upload service.
Handles video and image upload processing.
"""
import os
import uuid
import logging
from pathlib import Path
from flask import current_app, url_for

logger = logging.getLogger(__name__)


def handle_file_upload(file, model_name, counting_line_enabled=False):
    """
    Handle uploaded file (video or image).
    
    Args:
        file: FileStorage object
        model_name: Name of YOLO model to use
        counting_line_enabled: Enable counting line for videos
        
    Returns:
        dict: Result with status and details
    """
    try:
        # Get file extension
        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Validate file type
        if not _is_valid_file(file_ext):
            return {
                'status': 'error',
                'message': f'Unsupported file type: {file_ext}',
                'status_code': 400
            }
        
        # Process based on file type
        if _is_video(file_ext):
            return _process_video_upload(file, file_ext, model_name, counting_line_enabled)
        elif _is_image(file_ext):
            return _process_image_upload(file, file_ext, model_name)
        else:
            return {
                'status': 'error',
                'message': 'Unknown file type',
                'status_code': 400
            }
            
    except Exception as e:
        logger.error(f"Error handling file upload: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': f'Upload failed: {str(e)}',
            'status_code': 500
        }


def _is_valid_file(file_ext):
    """Check if file extension is valid"""
    allowed_video = current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', {'mp4', 'avi', 'mov', 'mkv'})
    allowed_image = current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', {'jpg', 'jpeg', 'png', 'bmp', 'gif'})
    return file_ext.lstrip('.') in (allowed_video | allowed_image)


def _is_video(file_ext):
    """Check if file is a video"""
    allowed_video = current_app.config.get('ALLOWED_VIDEO_EXTENSIONS', {'mp4', 'avi', 'mov', 'mkv'})
    return file_ext.lstrip('.') in allowed_video


def _is_image(file_ext):
    """Check if file is an image"""
    allowed_image = current_app.config.get('ALLOWED_IMAGE_EXTENSIONS', {'jpg', 'jpeg', 'png', 'bmp', 'gif'})
    return file_ext.lstrip('.') in allowed_image


def _process_video_upload(file, file_ext, model_name, counting_line_enabled):
    """Process video upload"""
    from database import save_video_info
    from app.services.video_service import initialize_video_processing
    
    try:
        # Generate unique video ID
        video_id = str(uuid.uuid4())
        video_filename = f'video_{video_id}{file_ext}'
        
        # Save video file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        video_path = os.path.join(upload_folder, video_filename)
        
        file.save(video_path)
        logger.info(f"Video saved: {video_path} (ID: {video_id})")
        
        # Save video info to database
        save_video_info(video_id, video_path, counting_line_enabled)
        
        # Initialize video processing
        initialize_video_processing(
            video_id=video_id,
            video_path=video_path,
            model_name=model_name,
            counting_line_enabled=counting_line_enabled
        )
        
        return {
            'status': 'success',
            'message': 'Video uploaded successfully!',
            'video_id': video_id,
            'status_code': 200
        }
        
    except Exception as e:
        logger.error(f"Error processing video upload: {e}", exc_info=True)
        raise


def _process_image_upload(file, file_ext, model_name):
    """Process image upload"""
    from database import save_video_info
    from app.services.image_service import process_image
    
    try:
        # Generate unique image ID
        image_id = str(uuid.uuid4())
        image_filename = f'{image_id}{file_ext}'
        
        # Save image file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        image_path = os.path.join(upload_folder, image_filename)
        
        file.save(image_path)
        logger.info(f"Image saved: {image_path} (ID: {image_id})")
        
        # Create video_id for image
        video_id = f'image_upload_{image_id}'
        
        # Save image info
        save_video_info(video_id=video_id, video_path=image_path, counting_line_enabled=False)
        
        # Process image
        try:
            result = process_image(
                image_path=image_path,
                video_id=video_id,
                model_name=model_name
            )
        except RuntimeError as e:
            logger.error(f"Image processing failed: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'status_code': 500
            }
        
        return {
            'status': 'success',
            'message': 'Image processed successfully!',
            'processed_image': url_for('main.uploaded_file', filename=os.path.basename(result['processed_path'])),
            'class_type': result.get('class_type'),
            'status_code': 200
        }
        
    except Exception as e:
        logger.error(f"Error processing image upload: {e}", exc_info=True)
        raise

