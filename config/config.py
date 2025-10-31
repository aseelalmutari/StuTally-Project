# config/config.py
"""
Configuration management for StuTally application.
Loads settings from environment variables with fallback defaults.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class"""
    
    # ===================================
    # FLASK CONFIGURATION
    # ===================================
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-CHANGE-IN-PRODUCTION')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # ===================================
    # JWT CONFIGURATION
    # ===================================
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'dev-jwt-secret-CHANGE-IN-PRODUCTION')
    JWT_ACCESS_TOKEN_EXPIRES = int(os.getenv('JWT_ACCESS_TOKEN_EXPIRES', 3600))
    
    # ===================================
    # ROBOFLOW API CONFIGURATION
    # ===================================
    ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')
    ROBOFLOW_MODEL_ID = os.getenv('ROBOFLOW_MODEL_ID', 'ss-uniform/3')
    
    # ===================================
    # DIRECTORY PATHS
    # ===================================
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    MODELS_FOLDER = BASE_DIR / 'models'
    DATA_FOLDER = BASE_DIR / 'data'
    STATIC_FOLDER = BASE_DIR / 'static'
    PROCESSED_IMAGES = STATIC_FOLDER / 'processed_images'
    LOGS_FOLDER = BASE_DIR / 'logs'
    TEMPLATES_FOLDER = BASE_DIR / 'templates'
    
    # ===================================
    # DATABASE CONFIGURATION
    # ===================================
    DATABASE_TYPE = os.getenv('DATABASE_TYPE', 'sqlite')
    SQLITE_PATH = os.getenv('SQLITE_PATH', 'data/detections.db')
    DATABASE_PATH = str(BASE_DIR / SQLITE_PATH)
    
    # PostgreSQL settings (for future migration)
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'stutally')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'stutally_db')
    
    @property
    def SQLALCHEMY_DATABASE_URI(self):
        """Generate database URI based on type"""
        if self.DATABASE_TYPE == 'postgresql':
            return (
                f'postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}'
                f'@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}'
            )
        else:
            return f'sqlite:///{self.DATABASE_PATH}'
    
    # ===================================
    # MODEL CONFIGURATION
    # ===================================
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'yolov8s.pt')
    FRAME_SKIP = int(os.getenv('FRAME_SKIP', 3))
    DEVICE = os.getenv('DEVICE', 'auto')  # 'cuda', 'cpu', or 'auto'
    MODEL_CACHE_SIZE = int(os.getenv('MODEL_CACHE_SIZE', 3))
    
    # Available YOLO models with download URLs
    YOLO_MODELS = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
        'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
        'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt',
        'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt',
    }
    
    # Custom model class names mapping
    CUSTOM_MODEL_CLASS_NAMES = {
        "High": "High School",
        "Middle": "Middle School",
        "students": "Unknown"
    }
    
    # ===================================
    # DEEPSORT CONFIGURATION
    # ===================================
    DEEPSORT_MAX_AGE = int(os.getenv('DEEPSORT_MAX_AGE', 15))
    DEEPSORT_N_INIT = int(os.getenv('DEEPSORT_N_INIT', 3))
    DEEPSORT_MAX_IOU = float(os.getenv('DEEPSORT_MAX_IOU', 0.7))
    
    # ===================================
    # SERVER CONFIGURATION
    # ===================================
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 5000))
    
    # ===================================
    # SOCKETIO CONFIGURATION
    # ===================================
    SOCKETIO_ASYNC_MODE = os.getenv('SOCKETIO_ASYNC_MODE', 'eventlet')
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.getenv('SOCKETIO_CORS_ALLOWED_ORIGINS', '*')
    
    # ===================================
    # ANALYTICS CONFIGURATION
    # ===================================
    ANALYTICS_UPDATE_INTERVAL = int(os.getenv('ANALYTICS_UPDATE_INTERVAL', 30))
    
    # ===================================
    # LOGGING CONFIGURATION
    # ===================================
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/app.log')
    LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', 10485760))  # 10MB
    LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', 10))
    
    # ===================================
    # FILE UPLOAD CONFIGURATION
    # ===================================
    MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', 500))
    MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024  # Convert to bytes
    
    ALLOWED_VIDEO_EXTENSIONS = set(
        os.getenv('ALLOWED_VIDEO_EXTENSIONS', 'mp4,avi,mov,mkv').split(',')
    )
    ALLOWED_IMAGE_EXTENSIONS = set(
        os.getenv('ALLOWED_IMAGE_EXTENSIONS', 'jpg,jpeg,png,bmp,gif').split(',')
    )
    
    @classmethod
    def init_app(cls, app=None):
        """
        Initialize application with configuration.
        Create necessary directories if they don't exist.
        """
        # Create directories
        directories = [
            cls.UPLOAD_FOLDER,
            cls.MODELS_FOLDER,
            cls.DATA_FOLDER,
            cls.PROCESSED_IMAGES,
            cls.LOGS_FOLDER
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        if app:
            # Validate configuration if app is provided
            cls.validate_config(app)
    
    @classmethod
    def validate_config(cls, app):
        """Validate configuration values"""
        # Warn if using default secrets in production
        if not cls.DEBUG and cls.SECRET_KEY == 'dev-secret-key-CHANGE-IN-PRODUCTION':
            app.logger.warning(
                "⚠️  WARNING: Using default SECRET_KEY in production! "
                "Set SECRET_KEY environment variable."
            )
        
        if not cls.DEBUG and cls.JWT_SECRET_KEY == 'dev-jwt-secret-CHANGE-IN-PRODUCTION':
            app.logger.warning(
                "⚠️  WARNING: Using default JWT_SECRET_KEY in production! "
                "Set JWT_SECRET_KEY environment variable."
            )
        
        # Warn if Roboflow API key is missing
        if not cls.ROBOFLOW_API_KEY:
            app.logger.warning(
                "⚠️  WARNING: ROBOFLOW_API_KEY not set. "
                "Custom model (best.pt) classification will not work."
            )


class DevelopmentConfig(Config):
    """Development environment configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class ProductionConfig(Config):
    """Production environment configuration"""
    DEBUG = False
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_config(cls, app):
        """Additional validation for production"""
        super().validate_config(app)
        
        # Require environment variables in production
        required_vars = ['SECRET_KEY', 'JWT_SECRET_KEY']
        missing = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or value.startswith('dev-'):
                missing.append(var)
        
        if missing:
            raise RuntimeError(
                f"❌ Missing required environment variables for production: {', '.join(missing)}\n"
                f"Please set these in your .env file."
            )


class TestingConfig(Config):
    """Testing environment configuration"""
    TESTING = True
    DEBUG = True
    DATABASE_PATH = ':memory:'  # Use in-memory SQLite for tests
    LOG_LEVEL = 'DEBUG'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env_name=None):
    """
    Get configuration object based on environment name.
    
    Args:
        env_name: Environment name ('development', 'production', 'testing')
                 If None, reads from FLASK_ENV environment variable
    
    Returns:
        Configuration class
    """
    if env_name is None:
        env_name = os.getenv('FLASK_ENV', 'development')
    
    return config.get(env_name, config['default'])

