#!/usr/bin/env python3
# scripts/test_full_system.py
"""
Comprehensive test script for Week 1 + Week 2 implementation.
Tests configuration, models, app structure, and core functionality.

Usage:
    python3 scripts/test_full_system.py
"""
import sys
import os
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("🧪 StuTally - Full System Test (Week 1 + Week 2)")
print("=" * 70)
print()

test_results = {
    'passed': 0,
    'failed': 0,
    'warnings': 0
}

def test_pass(message):
    """Mark test as passed"""
    global test_results
    test_results['passed'] += 1
    print(f"✅ {message}")

def test_fail(message):
    """Mark test as failed"""
    global test_results
    test_results['failed'] += 1
    print(f"❌ {message}")

def test_warn(message):
    """Mark test as warning"""
    global test_results
    test_results['warnings'] += 1
    print(f"⚠️  {message}")


# ============================================================================
# TEST SUITE 1: Environment & Configuration
# ============================================================================
print("📋 Test Suite 1: Environment & Configuration")
print("-" * 70)

# Test 1.1: .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['SECRET_KEY', 'JWT_SECRET_KEY', 'DEFAULT_MODEL']
    missing = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        test_warn(f".env file missing variables: {', '.join(missing)}")
    else:
        test_pass("Environment variables loaded")
        
except ImportError:
    test_fail("python-dotenv not installed: pip install python-dotenv")
except Exception as e:
    test_fail(f"Environment setup failed: {e}")

# Test 1.2: Configuration system
try:
    from config.config import Config, get_config
    
    config = get_config('development')
    test_pass(f"Configuration loaded: {config.__name__}")
    
    # Check essential config
    if config.SECRET_KEY and config.SECRET_KEY != 'dev-secret-key-CHANGE-IN-PRODUCTION':
        test_pass("SECRET_KEY configured properly")
    else:
        test_warn("SECRET_KEY using default (change for production)")
        
except Exception as e:
    test_fail(f"Configuration test failed: {e}")

print()


# ============================================================================
# TEST SUITE 2: Model Management (Week 1)
# ============================================================================
print("📋 Test Suite 2: Model Management")
print("-" * 70)

# Test 2.1: Model registry initialization
try:
    from ml.model_registry import init_model_registry
    from config.config import Config
    
    registry = init_model_registry(
        models_dir=Config.MODELS_FOLDER,
        device='auto',
        cache_size=3
    )
    
    test_pass(f"Model registry initialized (device: {registry.device})")
    
    if registry.available_models:
        test_pass(f"Found {len(registry.available_models)} models")
        for model_name in list(registry.available_models.keys())[:3]:
            print(f"      - {model_name}")
    else:
        test_warn("No models found in models/ folder")
        
except Exception as e:
    test_fail(f"Model registry failed: {e}")

# Test 2.2: Model caching performance
try:
    if registry.available_models:
        test_model = Config.DEFAULT_MODEL
        
        if test_model in registry.available_models:
            # First load
            start = time.time()
            model = registry.get_model(test_model)
            time1 = time.time() - start
            
            if model:
                test_pass(f"Model loaded: {test_model} ({time1:.3f}s)")
                
                # Cache test
                start = time.time()
                model2 = registry.get_model(test_model)
                time2 = time.time() - start
                
                speedup = time1 / time2 if time2 > 0 else 0
                if speedup > 100:
                    test_pass(f"Cache working: {speedup:.0f}x faster ({time2:.4f}s)")
                else:
                    test_warn(f"Cache improvement lower than expected: {speedup:.0f}x")
            else:
                test_fail(f"Model loading returned None")
        else:
            test_warn(f"Default model not found: {test_model}")
    else:
        print("   ⏩ Skipping model tests (no models available)")
        
except Exception as e:
    test_fail(f"Model caching test failed: {e}")

print()


# ============================================================================
# TEST SUITE 3: Application Structure (Week 2)
# ============================================================================
print("📋 Test Suite 3: Application Structure")
print("-" * 70)

# Test 3.1: App factory
try:
    from app import create_app
    
    app = create_app('development')
    test_pass(f"App factory created: {app.name}")
    
except Exception as e:
    test_fail(f"App factory failed: {e}")

# Test 3.2: Blueprints registered
try:
    blueprints = list(app.blueprints.keys())
    
    required_blueprints = ['main', 'analytics', 'api', 'auth']
    missing_bp = [bp for bp in required_blueprints if bp not in blueprints]
    
    if not missing_bp:
        test_pass(f"All blueprints registered: {', '.join(blueprints)}")
    else:
        test_fail(f"Missing blueprints: {', '.join(missing_bp)}")
        
except Exception as e:
    test_fail(f"Blueprint test failed: {e}")

# Test 3.3: Routes registered
try:
    routes = [rule.rule for rule in app.url_map.iter_rules()]
    
    key_routes = ['/', '/health', '/analytics/', '/api/model_status', '/login']
    missing_routes = [r for r in key_routes if r not in routes]
    
    if not missing_routes:
        test_pass(f"Key routes registered ({len(routes)} total)")
    else:
        test_fail(f"Missing routes: {', '.join(missing_routes)}")
        
except Exception as e:
    test_fail(f"Routes test failed: {e}")

# Test 3.4: Extensions (no circular imports)
try:
    from app.extensions import socketio, login_manager, bcrypt, jwt
    
    test_pass("Extensions imported (no circular imports)")
    
except Exception as e:
    test_fail(f"Extensions test failed (circular import?): {e}")

print()


# ============================================================================
# TEST SUITE 4: Services Layer
# ============================================================================
print("📋 Test Suite 4: Services Layer")
print("-" * 70)

# Test 4.1: Services import
try:
    from app.services.upload_service import handle_file_upload
    from app.services.video_service import initialize_video_processing
    from app.services.image_service import process_image
    from app.services.analytics_service import get_filtered_analytics
    from app.services.export_service import export_csv, export_pdf
    from app.services.realtime_service import get_latest_analytics
    
    test_pass("All services import successfully")
    
except ImportError as e:
    test_fail(f"Service import failed: {e}")
except Exception as e:
    test_warn(f"Service test warning: {e}")

print()


# ============================================================================
# TEST SUITE 5: Database
# ============================================================================
print("📋 Test Suite 5: Database")
print("-" * 70)

# Test 5.1: Database initialization
try:
    from database import init_db
    
    init_db()
    test_pass("Database initialized")
    
except Exception as e:
    test_fail(f"Database initialization failed: {e}")

# Test 5.2: Database path
try:
    db_path = Config.DATABASE_PATH
    if Path(db_path).exists():
        size = Path(db_path).stat().st_size / 1024  # KB
        test_pass(f"Database exists: {db_path} ({size:.1f} KB)")
    else:
        test_warn("Database file not found (will be created on first use)")
        
except Exception as e:
    test_warn(f"Database path check: {e}")

print()


# ============================================================================
# TEST SUITE 6: File Structure
# ============================================================================
print("📋 Test Suite 6: File Structure")
print("-" * 70)

# Test 6.1: Required directories
required_dirs = [
    Config.UPLOAD_FOLDER,
    Config.MODELS_FOLDER,
    Config.DATA_FOLDER,
    Config.LOGS_FOLDER,
    Config.PROCESSED_IMAGES
]

for directory in required_dirs:
    if directory.exists():
        test_pass(f"{directory.name}/ exists")
    else:
        test_warn(f"{directory.name}/ missing (will be created)")

# Test 6.2: Required files
required_files = [
    'run.py',
    'app/__init__.py',
    'app/extensions.py',
    'app/routes/__init__.py',
    'app/services/__init__.py',
    'config/config.py',
    'ml/model_registry.py',
    '.env.example'
]

for file_path in required_files:
    full_path = project_root / file_path
    if full_path.exists():
        test_pass(f"{file_path} exists")
    else:
        test_fail(f"{file_path} MISSING")

print()


# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("=" * 70)
print("📊 Test Results Summary")
print("=" * 70)
print()

total_tests = test_results['passed'] + test_results['failed'] + test_results['warnings']

print(f"Total Tests: {total_tests}")
print(f"✅ Passed:   {test_results['passed']}")
print(f"❌ Failed:   {test_results['failed']}")
print(f"⚠️  Warnings: {test_results['warnings']}")
print()

# Calculate score
if total_tests > 0:
    score = (test_results['passed'] / total_tests) * 100
    print(f"Score: {score:.1f}%")
    print()
    
    if score >= 90:
        print("🎉 Excellent! System is working great!")
        print("   ✅ All Week 1 + Week 2 features operational")
        print("   ✅ Ready for production use")
        print()
    elif score >= 70:
        print("✅ Good! System is mostly working")
        print("   ⚠️  Review warnings and fix failed tests")
        print("   📝 See TESTING_GUIDE.md for details")
        print()
    elif score >= 50:
        print("⚠️  Partial! Some features need attention")
        print("   ❌ Fix failed tests before production")
        print("   📝 Check WEEK2_MIGRATION.md for help")
        print()
    else:
        print("❌ Issues detected! System needs work")
        print("   🔧 Review all failed tests")
        print("   📚 Check documentation: TESTING_GUIDE.md")
        print()

# Next steps
print("=" * 70)
print("🎯 Next Steps")
print("=" * 70)
print()

if test_results['failed'] == 0:
    print("All tests passed! You can:")
    print("1. Test manually with UI: python3 run.py")
    print("2. Deploy to production")
    print("3. Continue to Week 3 when ready")
else:
    print("Fix failed tests first:")
    print("1. Review error messages above")
    print("2. Check TESTING_GUIDE.md for solutions")
    print("3. Re-run tests after fixes")

print()
print("=" * 70)

