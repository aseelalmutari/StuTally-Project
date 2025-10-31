# app/extensions.py
"""
Flask extensions initialization.
Centralizes all extensions to avoid circular imports.

This file creates extension instances that are imported by both
the app factory and route blueprints, preventing circular dependencies.
"""
from flask_socketio import SocketIO
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager

# Initialize extensions (but don't bind to app yet)
# These will be initialized in init_extensions() function

# Initialize without forcing async mode; engine will choose best available (threading now that eventlet is removed)
socketio = SocketIO()
login_manager = LoginManager()
bcrypt = Bcrypt()
jwt = JWTManager()


def init_extensions(app):
    """
    Initialize Flask extensions with app instance.
    Call this from create_app() factory.
    
    Args:
        app: Flask application instance
    """
    
    # SocketIO: let engine choose async mode (threading expected)
    socketio.init_app(
        app,
        cors_allowed_origins=app.config.get('SOCKETIO_CORS_ALLOWED_ORIGINS', '*'),
        logger=True,
        engineio_logger=False
    )
    
    # Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    
    # Bcrypt
    bcrypt.init_app(app)
    
    # JWT
    jwt.init_app(app)
    
    app.logger.info("✅ Flask extensions initialized")
    
    return app

