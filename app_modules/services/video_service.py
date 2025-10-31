# app/services/video_service.py
"""
Video processing service.
Handles video streaming and frame processing with YOLO + DeepSort.
"""
import cv2
import logging
import threading
import os
from typing import Generator, Optional
from flask import current_app
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

# Global state for video processing
_current_video_id = None
_current_video_path = None
_line_enabled = False
_line_position = None
_tracker = None
_tracker_lock = threading.Lock()
_tracked_ids = set()
_previous_positions = {}
_entry_counts = {}
_exit_counts = {}
_total_counts = {}
_track_labels = {}


def initialize_video_processing(video_id: str, video_path: str, model_name: str, counting_line_enabled: bool):
    """
    Initialize video processing state.
    
    Args:
        video_id: Unique video identifier
        video_path: Path to video file
        model_name: YOLO model to use
        counting_line_enabled: Enable counting line
    """
    global _current_video_id, _current_video_path, _line_enabled
    global _tracker, _tracked_ids, _entry_counts, _exit_counts, _total_counts
    global _previous_positions, _track_labels
    
    logger.info(f"Initializing video processing for: {video_id}")
    logger.info(f"Video path: {video_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Counting line enabled: {counting_line_enabled}")
    
    _current_video_id = video_id
    _current_video_path = video_path
    _line_enabled = counting_line_enabled
    
    # Reset tracking state
    with _tracker_lock:
        config = current_app.config
        
        try:
            # Initialize DeepSort tracker
            logger.info("Initializing DeepSort tracker...")
            _tracker = DeepSort(
                max_age=config.get('DEEPSORT_MAX_AGE', 15),
                n_init=config.get('DEEPSORT_N_INIT', 3),
                max_iou_distance=config.get('DEEPSORT_MAX_IOU', 0.7)
            )
            logger.info("DeepSort tracker initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize DeepSort tracker: {e}", exc_info=True)
            logger.warning("Continuing without DeepSort tracker - video will work but without tracking")
            _tracker = None
        
        _tracked_ids = set()
        _entry_counts = {}
        _exit_counts = {}
        _total_counts = {}
        _previous_positions = {}
        _track_labels = {}
    
    logger.info(f"Video processing initialized: {video_id}")


def get_video_stream(video_id: str) -> Optional[Generator]:
    """
    Get video stream generator for a video ID.
    
    Args:
        video_id: Video identifier
        
    Returns:
        Generator yielding video frames or None
    """
    global _current_video_id, _current_video_path
    
    logger.info(f"Requesting video stream for ID: {video_id}")
    logger.info(f"Current video ID: {_current_video_id}")
    logger.info(f"Current video path: {_current_video_path}")
    
    if video_id != _current_video_id:
        logger.warning(f"Video ID mismatch: {video_id} != {_current_video_id}")
        return None
    
    if not _current_video_path or not os.path.exists(_current_video_path):
        logger.warning(f"Video path not found: {_current_video_path}")
        return None
    
    logger.info(f"Starting video stream for: {video_id}")
    return _generate_frames(video_id, _current_video_path)


def _generate_frames(video_id: str, video_path: str) -> Generator:
    """
    Generate processed video frames.
    
    Args:
        video_id: Video identifier
        video_path: Path to video file
        
    Yields:
        Encoded frame bytes in MJPEG format
    """
    from ml.model_loader import get_current_model
    from database import log_detection
    from flask import current_app
    
    global _current_video_id, _line_enabled, _line_position
    global _tracker, _tracked_ids, _entry_counts, _exit_counts, _total_counts
    global _previous_positions, _track_labels
    
    cap = cv2.VideoCapture(video_path)
    
    # Get config values outside of app context to avoid context issues
    frame_skip = 3  # Default frame skip
    default_model = 'yolov8s.pt'  # Default model
    device = 'cpu'  # Default device
    
    # Try to get config from current app if available
    try:
        if current_app:
            frame_skip = current_app.config.get('FRAME_SKIP', 3)
            default_model = current_app.config.get('DEFAULT_MODEL', 'yolov8s.pt')
            device = current_app.config.get('DEVICE', 'cpu')
    except RuntimeError:
        # Working outside app context, use defaults
        logger.warning("Working outside app context, using default config values")
    
    frame_count = 0
    
    logger.info(f"Opening video: {video_path}")
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    logger.info(f"Video opened successfully: {video_path}")
    
    # Get frame dimensions and set line position
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _line_position = (0, frame_height // 2, frame_width, frame_height // 2)
    
    logger.info(f"Video dimensions: {frame_width}x{frame_height}")
    logger.info(f"Frame skip: {frame_skip}")
    
    # Get model
    model = get_current_model(default_model=default_model)
    if model is None:
        logger.error("No model available")
        cap.release()
        return
    
    try:
        logger.info(f"Starting frame generation loop for {video_id}")
        while True:
            # Check if video ID changed
            if video_id != _current_video_id:
                logger.info(f"Stopping video feed for {video_id}")
                break
            
            success, frame = cap.read()
            if not success:
                logger.info(f"End of video: {video_path}")
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            if frame_count % 30 == 0:  # Log every 30 frames
                logger.info(f"Processing frame {frame_count} for {video_id}")
            
            # Process frame
            processed_frame = _process_frame(
                frame=frame,
                model=model,
                video_id=video_id
            )
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret:
                logger.error("Failed to encode frame")
                continue
            
            # Yield frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                   
    except GeneratorExit:
        logger.info(f"Video stream closed for {video_id}")
    except Exception as e:
        logger.error(f"Error in frame generation: {e}", exc_info=True)
    finally:
        cap.release()
        logger.info(f"Video capture released for {video_id}")


def _process_frame(frame, model, video_id):
    """
    Process a single frame with YOLO + DeepSort.
    
    Args:
        frame: Video frame (numpy array)
        model: YOLO model instance
        video_id: Video identifier
        
    Returns:
        Processed frame with annotations
    """
    from database import log_detection
    import numpy as np
    
    global _tracker, _tracked_ids, _line_enabled, _line_position
    global _entry_counts, _exit_counts, _total_counts, _previous_positions, _track_labels
    
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting frame color: {e}")
        return frame
    
    # YOLO detection
    try:
        # Use 'cpu' to avoid CUDA errors and app context issues
        device = 'cpu'
        try:
            if current_app:
                device = current_app.config.get('DEVICE', 'cpu')
                if device == 'auto':
                    device = 'cpu'
        except RuntimeError:
            # Working outside app context, use default
            device = 'cpu'
            
        results = model.predict(source=frame_rgb, device=device, verbose=False)
        detections = results[0].boxes if results and len(results) > 0 else []
    except Exception as e:
        logger.error(f"YOLO prediction error: {e}")
        return frame
    
    # Prepare detections for DeepSort
    detections_list = []
    for box in detections:
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = model.names.get(class_id, "Unknown")
            
            # Filter by class (person or custom classes)
            if label == "person" or label in ["High", "Middle", "students"]:
                detections_list.append(([x1, y1, x2 - x1, y2 - y1], confidence, class_id, label, None))
        except Exception as e:
            logger.error(f"Detection processing error: {e}")
    
    # Update tracker
    with _tracker_lock:
        try:
            tracks = _tracker.update_tracks(detections_list, frame=frame)
        except Exception as e:
            logger.error(f"Tracker update error: {e}")
            tracks = []
        
        # Process tracks
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            y_center = (y1 + y2) // 2
            
            # Get label
            label = _track_labels.get(track_id, {}).get('label', 'person')
            stage = _track_labels.get(track_id, {}).get('stage', None)
            
            # Initialize counts
            if label not in _entry_counts:
                _entry_counts[label] = 0
                _exit_counts[label] = 0
                _total_counts[label] = 0
            
            color = (255, 255, 255)  # White by default
            
            # Line crossing detection
            if _line_enabled and _line_position:
                line_y = _line_position[1]
                
                if track_id in _previous_positions:
                    y_prev = _previous_positions[track_id]
                    direction = _check_line_crossing(y_prev, y_center, line_y)
                    
                    if direction == "down" and track_id not in _tracked_ids:
                        _tracked_ids.add(track_id)
                        _exit_counts[label] += 1
                        log_detection(video_id, label, 0.9, x1, y1, x2, y2, track_id, stage, action='exit')
                        color = (0, 0, 255)  # Red
                        logger.debug(f"Track {track_id} exited")
                    elif direction == "up" and track_id not in _tracked_ids:
                        _tracked_ids.add(track_id)
                        _entry_counts[label] += 1
                        log_detection(video_id, label, 0.9, x1, y1, x2, y2, track_id, stage, action='entry')
                        color = (0, 255, 0)  # Green
                        logger.debug(f"Track {track_id} entered")
                
                _previous_positions[track_id] = y_center
                
                # Draw counting line
                cv2.line(frame, (_line_position[0], _line_position[1]),
                        (_line_position[2], _line_position[3]), (0, 0, 255), 2)
            else:
                # No line - just count unique tracks
                if track_id not in _tracked_ids:
                    _tracked_ids.add(track_id)
                    _total_counts[label] += 1
                    log_detection(video_id, label, 0.9, x1, y1, x2, y2, track_id, stage, action='counted')
                    logger.debug(f"Track {track_id} counted")
            
            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {track_id}', (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display counters
        y_offset = 30
        if _line_enabled:
            for label in _entry_counts:
                cv2.putText(frame, f'Entries ({label}): {_entry_counts[label]}',
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset += 30
            for label in _exit_counts:
                cv2.putText(frame, f'Exits ({label}): {_exit_counts[label]}',
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                y_offset += 30
        else:
            for label in _total_counts:
                cv2.putText(frame, f'Total ({label}): {_total_counts[label]}',
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                y_offset += 30
    
    return frame


def _check_line_crossing(y_previous: int, y_current: int, line_y: int) -> Optional[str]:
    """
    Check if object crossed the counting line.
    
    Args:
        y_previous: Previous Y position
        y_current: Current Y position
        line_y: Line Y coordinate
        
    Returns:
        'down' for downward crossing, 'up' for upward, None otherwise
    """
    if y_previous < line_y and y_current >= line_y:
        return "down"
    elif y_previous > line_y and y_current <= line_y:
        return "up"
    return None


import os  # Add this import at the top

