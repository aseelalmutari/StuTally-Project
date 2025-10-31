# app/services/image_service.py
"""
Image processing service.
Handles image detection with YOLO or custom Roboflow model.
"""
import cv2
import logging
import os
from flask import current_app

logger = logging.getLogger(__name__)


def process_image(image_path: str, video_id: str, model_name: str) -> dict:
    """
    Process image with selected model.
    
    Args:
        image_path: Path to image file
        video_id: Unique identifier
        model_name: Model to use ('best.pt' for custom, else YOLO)
        
    Returns:
        dict with processed_path and class_type (if custom model)
    """
    if model_name == 'best.pt':
        return _process_with_custom_model(image_path, video_id)
    else:
        return _process_with_yolo(image_path, video_id, model_name)


def _process_with_custom_model(image_path: str, video_id: str) -> dict:
    """Process image with Roboflow custom model"""
    # Roboflow inference is optional; guard import to avoid hard crash when package is missing
    try:
        from inference import get_model
    except Exception as import_error:
        raise RuntimeError("Roboflow inference package is not installed. Set ROBOFLOW_API_KEY or install 'inference' package.") from import_error
    import supervision as sv
    from database import log_detection
    
    api_key = current_app.config.get('ROBOFLOW_API_KEY')
    model_id = current_app.config.get('ROBOFLOW_MODEL_ID', 'ss-uniform/3')
    
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY not configured")
    
    logger.info("Processing image with custom model")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load model
    model = get_model(model_id=model_id, api_key=api_key)
    
    # Inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)
    
    # Annotate
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    
    # Determine class type
    id_to_class_name = {0: "High", 1: "Middle", 2: "students"}
    class_names = [id_to_class_name.get(cid, "Unknown") for cid in detections.class_id]
    unique_classes = set(class_names)
    
    if "High" in unique_classes and "Middle" in unique_classes:
        class_type = "High School & Middle School"
    elif "High" in unique_classes:
        class_type = "High School"
    elif "Middle" in unique_classes:
        class_type = "Middle School"
    else:
        class_type = "Unknown"
    
    # Add text to image
    cv2.putText(annotated_image, f"Educational Level: {class_type}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save processed image
    processed_path = os.path.splitext(image_path)[0] + '_processed.jpg'
    cv2.imwrite(processed_path, annotated_image)
    
    # Log detections
    custom_model_class_names = current_app.config.get('CUSTOM_MODEL_CLASS_NAMES', {})
    for class_id, confidence, box in zip(detections.class_id, detections.confidence, detections.xyxy):
        class_name = id_to_class_name.get(class_id, "Unknown")
        stage = custom_model_class_names.get(class_name, "Unknown") if class_name != "students" else "Unknown"
        x1, y1, x2, y2 = map(int, box)
        log_detection(video_id, class_name, confidence, x1, y1, x2, y2, 0, stage, action='counted')
    
    logger.info(f"Custom model processing complete: {class_type}")
    
    return {
        'processed_path': processed_path,
        'class_type': class_type
    }


def _process_with_yolo(image_path: str, video_id: str, model_name: str) -> dict:
    """Process image with standard YOLO model"""
    from ml.model_loader import get_model_sync
    from database import log_detection
    
    logger.info(f"Processing image with YOLO model: {model_name}")
    
    # Load model
    model = get_model_sync(model_name)
    if model is None:
        raise ValueError(f"Failed to load model: {model_name}")
    
    # Run prediction
    device = current_app.config.get('DEVICE', 'auto')
    results = model.predict(source=image_path, device=device, conf=0.5)
    
    if not results or len(results) == 0:
        raise ValueError("No detections found")
    
    # Get annotated frame
    annotated_frame = results[0].plot()
    
    # Add text
    stage = "Unknown"
    cv2.putText(annotated_frame, f"Educational Level: {stage}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Save processed image
    processed_path = os.path.splitext(image_path)[0] + '_processed.jpg'
    cv2.imwrite(processed_path, annotated_frame)
    
    # Log detections
    detections = results[0].boxes
    for box in detections:
        class_id = int(box.cls[0])
        class_name = model.names.get(class_id, "Unknown")
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        log_detection(video_id, class_name, confidence, x1, y1, x2, y2, 0, None, action='counted')
    
    logger.info("YOLO processing complete")
    
    return {
        'processed_path': processed_path,
        'class_type': None
    }

