# ml/model_loader.py
"""
High-level model loading service.
Provides simple interface for model switching and management.
"""
from typing import Optional, Dict
from ml.model_registry import get_registry, init_model_registry as _init_registry
from config.config import Config
import logging

logger = logging.getLogger(__name__)


def init_model_service(app=None) -> None:
    """
    Initialize the model service with configuration.
    Call this during Flask app initialization.
    
    Args:
        app: Flask application instance (optional)
    """
    if app:
        models_dir = app.config.get('MODELS_FOLDER', Config.MODELS_FOLDER)
        device = app.config.get('DEVICE', Config.DEVICE)
        cache_size = app.config.get('MODEL_CACHE_SIZE', Config.MODEL_CACHE_SIZE)
        default_model = app.config.get('DEFAULT_MODEL', Config.DEFAULT_MODEL)
    else:
        models_dir = Config.MODELS_FOLDER
        device = Config.DEVICE
        cache_size = Config.MODEL_CACHE_SIZE
        default_model = Config.DEFAULT_MODEL
    
    # Initialize registry
    registry = _init_registry(
        models_dir=models_dir,
        device=device,
        cache_size=cache_size,
        preload_models=[default_model]  # Preload default model
    )
    
    logger.info(f"✅ Model service initialized with default model: {default_model}")
    
    return registry


async def switch_model_async(model_name: str) -> Dict:
    """
    Switch to a different model asynchronously.
    This is non-blocking and returns immediately with status.
    
    Args:
        model_name: Name of model file (e.g., 'yolov8s.pt')
        
    Returns:
        Dictionary with status and message
    """
    registry = get_registry()
    
    try:
        # Check if model exists
        if model_name not in registry.available_models and model_name not in registry._get_downloadable_models():
            return {
                'status': 'error',
                'model': model_name,
                'message': f'Model not found: {model_name}',
                'available_models': list(registry.available_models.keys())
            }
        
        # Load model asynchronously
        logger.info(f"🔄 Switching to model: {model_name}")
        model = await registry.load_model_async(model_name)
        
        if model:
            return {
                'status': 'success',
                'model': model_name,
                'message': f'Successfully switched to model: {model_name}',
                'cache_status': registry.get_cache_status()
            }
        else:
            return {
                'status': 'error',
                'model': model_name,
                'message': f'Failed to load model: {model_name}'
            }
            
    except Exception as e:
        logger.error(f"❌ Error switching to model '{model_name}': {e}", exc_info=True)
        return {
            'status': 'error',
            'model': model_name,
            'message': str(e)
        }


def get_model_sync(model_name: str):
    """
    Get a model synchronously (blocking).
    Use this in contexts where async is not available.
    
    Args:
        model_name: Name of model file
        
    Returns:
        YOLO model instance or None
    """
    registry = get_registry()
    return registry.get_model(model_name)


def get_current_model(default_model: str = None):
    """
    Get the most recently used model from cache.
    
    Args:
        default_model: Model to load if cache is empty
        
    Returns:
        YOLO model instance
    """
    registry = get_registry()
    cache_status = registry.get_cache_status()
    
    if cache_status['cached_models']:
        # Get most recently used model (last in cache order)
        recent_model_name = cache_status['cache_order'][-1]
        return registry.get_model(recent_model_name)
    elif default_model:
        # Load default model if cache is empty
        logger.info(f"Cache empty, loading default model: {default_model}")
        return registry.get_model(default_model)
    else:
        return None


def get_cache_info() -> Dict:
    """
    Get information about model cache.
    
    Returns:
        Dictionary with cache status
    """
    registry = get_registry()
    return registry.get_cache_status()


def clear_model_cache() -> Dict:
    """
    Clear all models from cache and free memory.
    
    Returns:
        Dictionary with result status
    """
    try:
        registry = get_registry()
        registry.clear_cache()
        return {
            'status': 'success',
            'message': 'Model cache cleared successfully'
        }
    except Exception as e:
        logger.error(f"❌ Error clearing model cache: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }


def refresh_models() -> Dict:
    """
    Rescan models directory for new models.
    
    Returns:
        Dictionary with available models
    """
    try:
        registry = get_registry()
        registry.refresh_available_models()
        return {
            'status': 'success',
            'message': 'Models refreshed successfully',
            'available_models': list(registry.available_models.keys())
        }
    except Exception as e:
        logger.error(f"❌ Error refreshing models: {e}", exc_info=True)
        return {
            'status': 'error',
            'message': str(e)
        }


def preload_common_models() -> None:
    """
    Preload commonly used models into cache.
    Call this during application startup for faster first-time inference.
    """
    registry = get_registry()
    
    # Common models to preload
    common_models = [
        Config.DEFAULT_MODEL,
        'yolov8n.pt',  # Fast model
        'best.pt'       # Custom model
    ]
    
    # Filter to only existing models
    existing_models = [
        model for model in common_models 
        if model in registry.available_models
    ]
    
    if existing_models:
        logger.info(f"🔥 Preloading common models: {existing_models}")
        registry.preload_models(existing_models)
    else:
        logger.warning("⚠️  No common models found to preload")

