#!/usr/bin/env python3
# run.py
"""
Application entry point.
Runs the Flask application with SocketIO.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

from app import create_app
from app.extensions import socketio


# Determine environment
env = os.getenv('FLASK_ENV', 'development')

# Create app
app = create_app(config_name=env)

if __name__ == '__main__':
    # Get configuration
    host = app.config.get('HOST', '0.0.0.0')
    port = app.config.get('PORT', 5001)  # Changed to 5001 to avoid AirPlay conflict
    debug = app.config.get('DEBUG', False)
    
    # Run with SocketIO
    app.logger.info(f"🚀 Starting server on {host}:{port}")
    app.logger.info(f"   Environment: {env}")
    app.logger.info(f"   Debug: {debug}")
    
    socketio.run(
        app,
        host=host,
        port=port,
        debug=debug,
        use_reloader=debug,
        allow_unsafe_werkzeug=True
    )

