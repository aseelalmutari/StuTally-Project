"""
StuTally Version Information
"""

__version__ = '2.0.0'
__version_info__ = (2, 0, 0)
__release_date__ = '2025-10-31'
__author__ = 'StuTally Contributors'
__description__ = 'Smart Student Tracking and Analytics System'
__url__ = 'https://github.com/aseelalmutari/StuTally-Project'
__license__ = 'MIT'

# Build information
__build__ = 'stable'
__python_requires__ = '>=3.8'

# Feature flags
FEATURES = {
    'yolo_detection': True,
    'deepsort_tracking': True,
    'custom_model': True,
    'realtime_streaming': True,
    'analytics_dashboard': True,
    'jwt_auth': True,
    'csv_export': True,
    'pdf_export': True,
    'heatmap': True,
    'stage_comparison': True,
}

# Model versions
MODEL_VERSIONS = {
    'yolov8': '8.0.0+',
    'custom': '3.0',
}

def get_version():
    """Return version string"""
    return __version__

def get_version_info():
    """Return version tuple"""
    return __version_info__

def print_version():
    """Print version information"""
    print(f"StuTally v{__version__}")
    print(f"Released: {__release_date__}")
    print(f"Python: {__python_requires__}")
    print(f"License: {__license__}")

if __name__ == '__main__':
    print_version()

