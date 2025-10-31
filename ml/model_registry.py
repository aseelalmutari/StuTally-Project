# ml/model_registry.py
"""
Thread-safe Model Registry with intelligent caching.
Solves the model switching delay problem by keeping models in memory.

Features:
- Async model loading (non-blocking)
- LRU cache for multiple models
- Thread-safe operations
- GPU memory management
- Model preloading on startup
"""
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, List
from pathlib import Path
import torch
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Thread-safe model registry with caching and async loading.
    Prevents model switching delays by keeping models in memory.
    
    Usage:
        registry = ModelRegistry(models_dir='models/', device='auto', cache_size=3)
        
        # Async loading (recommended)
        model = await registry.load_model_async('yolov8s.pt')
        
        # Sync loading (fallback)
        model = registry.get_model('yolov8s.pt')
    """
    
    def __init__(
        self,
        models_dir: Path,
        device: str = 'auto',
        cache_size: int = 3,
        download_models: bool = True
    ):
        """
        Initialize Model Registry.
        
        Args:
            models_dir: Directory containing model files
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
            cache_size: Maximum number of models to keep in memory
            download_models: Auto-download missing models
        """
        self.models_dir = Path(models_dir)
        self.device = self._get_device(device)
        self.cache_size = cache_size
        self.download_models = download_models
        
        # Model cache: {model_name: model_instance}
        self._cache: Dict[str, YOLO] = {}
        self._cache_lock = threading.RLock()
        self._cache_order: List[str] = []  # Track LRU order
        
        # Loading locks to prevent duplicate loading
        self._loading_locks: Dict[str, threading.Lock] = {}
        
        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='ModelLoader')
        
        # Scan available models
        self.available_models = self._scan_available_models()
        
        logger.info(f"✅ ModelRegistry initialized")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Cache size: {cache_size}")
        logger.info(f"   Available models: {list(self.available_models.keys())}")
    
    def _get_device(self, device: str) -> str:
        """Determine which device to use for inference"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        return device
    
    def _scan_available_models(self) -> Dict[str, Path]:
        """Scan models directory for available .pt files"""
        models = {}
        if self.models_dir.exists():
            for pt_file in self.models_dir.glob('*.pt'):
                models[pt_file.name] = pt_file
        else:
            logger.warning(f"⚠️  Models directory not found: {self.models_dir}")
            self.models_dir.mkdir(parents=True, exist_ok=True)
        
        return models
    
    def _load_model_sync(self, model_name: str) -> Optional[YOLO]:
        """
        Synchronously load a model (runs in thread pool).
        This is the heavy operation that blocks - we want to avoid blocking main thread.
        
        Args:
            model_name: Name of model file (e.g., 'yolov8s.pt')
            
        Returns:
            Loaded YOLO model or None if loading failed
        """
        try:
            model_path = self.available_models.get(model_name)
            
            if not model_path or not model_path.exists():
                logger.warning(f"⚠️  Model file not found: {model_name}")
                
                # Try to download if enabled
                if self.download_models and model_name in self._get_downloadable_models():
                    logger.info(f"📥 Attempting to download {model_name}...")
                    model_path = self._download_model(model_name)
                    if not model_path:
                        return None
                else:
                    return None
            
            logger.info(f"🔄 Loading model '{model_name}' on {self.device}...")
            start_time = time.time()
            
            # Load YOLO model
            model = YOLO(str(model_path))
            model.to(self.device)
            
            # Warm up model with dummy inference (speeds up first real inference)
            if self.device == 'cuda':
                logger.debug(f"   Warming up GPU for {model_name}...")
                dummy_input = torch.zeros(1, 3, 640, 640).to(self.device)
                try:
                    with torch.no_grad():
                        _ = model.predict(dummy_input, verbose=False)
                except Exception as warmup_error:
                    logger.warning(f"   GPU warmup failed: {warmup_error}")
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Model '{model_name}' loaded in {elapsed:.2f}s")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error loading model '{model_name}': {e}", exc_info=True)
            return None
    
    def _get_downloadable_models(self) -> Dict[str, str]:
        """Get dict of models that can be auto-downloaded"""
        # Import here to avoid circular import
        from config.config import Config
        return Config.YOLO_MODELS
    
    def _download_model(self, model_name: str) -> Optional[Path]:
        """
        Download a model from Ultralytics repository.
        
        Args:
            model_name: Name of model file
            
        Returns:
            Path to downloaded model or None if failed
        """
        try:
            import requests
            
            downloadable = self._get_downloadable_models()
            url = downloadable.get(model_name)
            
            if not url:
                logger.error(f"❌ No download URL found for {model_name}")
                return None
            
            model_path = self.models_dir / model_name
            
            logger.info(f"📥 Downloading {model_name} from {url}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 10MB
                        if total_size and downloaded % (10 * 1024 * 1024) < 8192:
                            progress = (downloaded / total_size) * 100
                            logger.info(f"   Download progress: {progress:.1f}%")
            
            logger.info(f"✅ Downloaded {model_name} successfully")
            
            # Update available models
            self.available_models[model_name] = model_path
            
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Failed to download model '{model_name}': {e}")
            return None
    
    async def load_model_async(self, model_name: str) -> Optional[YOLO]:
        """
        Asynchronously load a model in thread pool.
        Returns immediately if model is already cached.
        
        Args:
            model_name: Name of model file (e.g., 'yolov8s.pt')
            
        Returns:
            Loaded YOLO model or None if loading failed
        """
        # Check cache first (fast path)
        with self._cache_lock:
            if model_name in self._cache:
                logger.debug(f"✨ Model '{model_name}' retrieved from cache")
                # Move to end of LRU order
                self._cache_order.remove(model_name)
                self._cache_order.append(model_name)
                return self._cache[model_name]
        
        # Get or create lock for this model
        if model_name not in self._loading_locks:
            self._loading_locks[model_name] = threading.Lock()
        
        lock = self._loading_locks[model_name]
        
        # Check if another thread is already loading this model
        if lock.locked():
            logger.debug(f"⏳ Waiting for another thread to load '{model_name}'...")
            # Wait for the other thread to finish
            await asyncio.get_event_loop().run_in_executor(None, lock.acquire)
            lock.release()
            
            # Model should now be in cache
            with self._cache_lock:
                return self._cache.get(model_name)
        
        # Load model in thread pool (slow operation runs in background)
        with lock:
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                self._executor,
                self._load_model_sync,
                model_name
            )
            
            if model:
                with self._cache_lock:
                    # Add to cache
                    self._cache[model_name] = model
                    self._cache_order.append(model_name)
                    
                    # Evict oldest model if cache is full (LRU eviction)
                    if len(self._cache) > self.cache_size:
                        oldest_model = self._cache_order.pop(0)
                        logger.info(f"🗑️  Evicting model from cache: {oldest_model}")
                        del self._cache[oldest_model]
                        
                        # Free GPU memory if on CUDA
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()
                            logger.debug("   GPU cache cleared")
            
            return model
    
    def get_model(self, model_name: str) -> Optional[YOLO]:
        """
        Synchronously get a model (blocking).
        Use this in synchronous contexts where async is not available.
        
        Args:
            model_name: Name of model file
            
        Returns:
            Loaded YOLO model or None if loading failed
        """
        with self._cache_lock:
            if model_name in self._cache:
                # Move to end of LRU order
                self._cache_order.remove(model_name)
                self._cache_order.append(model_name)
                return self._cache[model_name]
        
        # Model not cached, load it synchronously
        model = self._load_model_sync(model_name)
        
        if model:
            with self._cache_lock:
                self._cache[model_name] = model
                self._cache_order.append(model_name)
                
                # Evict if needed
                if len(self._cache) > self.cache_size:
                    oldest_model = self._cache_order.pop(0)
                    del self._cache[oldest_model]
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
        
        return model
    
    def preload_models(self, model_names: List[str]):
        """
        Preload multiple models in background.
        Call this during app startup to warm up the cache.
        
        Args:
            model_names: List of model names to preload
        """
        logger.info(f"🔥 Preloading models: {model_names}")
        
        for model_name in model_names:
            # Submit to thread pool (non-blocking)
            self._executor.submit(self._load_model_sync, model_name)
    
    def clear_cache(self):
        """Clear model cache and free GPU memory"""
        with self._cache_lock:
            model_count = len(self._cache)
            
            for model_name in list(self._cache.keys()):
                del self._cache[model_name]
            
            self._cache_order.clear()
            
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"🗑️  Model cache cleared ({model_count} models removed)")
    
    def get_cache_status(self) -> dict:
        """
        Get current cache status.
        
        Returns:
            Dictionary with cache information
        """
        with self._cache_lock:
            return {
                'cached_models': list(self._cache.keys()),
                'cache_order': self._cache_order.copy(),
                'cache_size': len(self._cache),
                'max_cache_size': self.cache_size,
                'device': self.device,
                'available_models': list(self.available_models.keys())
            }
    
    def refresh_available_models(self):
        """Rescan models directory for new models"""
        self.available_models = self._scan_available_models()
        logger.info(f"🔄 Rescanned models directory: {list(self.available_models.keys())}")
    
    def __del__(self):
        """Cleanup on destruction"""
        try:
            self._executor.shutdown(wait=False)
            logger.debug("ModelRegistry executor shut down")
        except:
            pass


# Global registry instance (initialized later)
_model_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        ModelRegistry instance
        
    Raises:
        RuntimeError: If registry not initialized
    """
    if _model_registry is None:
        raise RuntimeError(
            "ModelRegistry not initialized. "
            "Call init_model_registry() first in your app setup."
        )
    return _model_registry


def init_model_registry(
    models_dir: Path,
    device: str = 'auto',
    cache_size: int = 3,
    preload_models: Optional[List[str]] = None
) -> ModelRegistry:
    """
    Initialize the global model registry.
    Call this during application startup.
    
    Args:
        models_dir: Directory containing model files
        device: 'cuda', 'cpu', or 'auto'
        cache_size: Maximum models to keep in cache
        preload_models: List of models to preload
        
    Returns:
        Initialized ModelRegistry instance
    """
    global _model_registry
    
    _model_registry = ModelRegistry(
        models_dir=models_dir,
        device=device,
        cache_size=cache_size
    )
    
    # Preload default models if specified
    if preload_models:
        _model_registry.preload_models(preload_models)
    
    logger.info("✅ Global model registry initialized")
    
    return _model_registry

