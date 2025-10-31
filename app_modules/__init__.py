# app/__init__.py
"""
Flask application factory.
Creates and configures the Flask application instance.
"""
from flask import Flask
import logging
from pathlib import Path


def create_app(config_name='development'):
    """
    Application factory pattern.
    Creates and configures Flask application.
    
    Args:
        config_name: Configuration name ('development', 'production', 'testing')
        
    Returns:
        Configured Flask application
    """
    # Create Flask app
    app = Flask(
        __name__,
        template_folder='../templates',
        static_folder='../static'
    )
    
    # Load configuration
    from config.config import get_config
    config_class = get_config(config_name)
    app.config.from_object(config_class)
    config_class.init_app(app)
    
    # Setup logging
    setup_logging(app)
    
    app.logger.info(f"🚀 Starting StuTally in {config_name} mode")
    
    # Initialize extensions
    from app.extensions import init_extensions
    init_extensions(app)
    
    # Initialize database
    from database import init_db
    init_db()
    app.logger.info("✅ Database initialized")
    
    # Initialize model service
    from ml.model_loader import init_model_service
    init_model_service(app)
    app.logger.info("✅ Model service initialized")
    
    # Register blueprints
    from app.routes import register_blueprints
    register_blueprints(app)
    
    # Initialize real-time service
    from app.services.realtime_service import init_realtime_service
    init_realtime_service(app)
    
    # Register user loader
    register_user_loader(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    app.logger.info("✅ Application initialization complete")
    
    return app


def setup_logging(app):
    """Configure application logging"""
    log_level = app.config.get('LOG_LEVEL', 'INFO')
    log_file = app.config.get('LOG_FILE', 'logs/app.log')
    
    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    app.logger.setLevel(getattr(logging, log_level))


def register_user_loader(app):
    """Register Flask-Login user loader"""
    from app.extensions import login_manager
    from auth import User
    import sqlite3
    
    @login_manager.user_loader
    def load_user(user_id):
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        row = conn.execute(
            "SELECT id, username, role FROM users WHERE id=?",
            (user_id,)
        ).fetchone()
        conn.close()
        return User(*row) if row else None


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def not_found(error):
        return {"error": "Not found"}, 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal error: {error}")
        return {"error": "Internal server error"}, 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        app.logger.error(f"Unhandled exception: {error}", exc_info=True)
        return {"error": "An unexpected error occurred"}, 500

