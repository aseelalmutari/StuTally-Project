# app/routes/__init__.py
"""
Routes package.
Registers all blueprint routes with the Flask app.
"""
from flask import Flask


def register_blueprints(app: Flask):
    """
    Register all application blueprints.
    
    Args:
        app: Flask application instance
    """
    from app.routes.main import main_bp
    from app.routes.analytics import analytics_bp
    from app.routes.api import api_bp
    from auth import auth_bp  # Existing auth blueprint
    
    # Register blueprints
    app.register_blueprint(main_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(auth_bp)
    
    app.logger.info("✅ All blueprints registered")

