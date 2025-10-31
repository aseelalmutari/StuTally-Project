# 🤝 Contributing Guide to StuTally

Thank you for your interest in contributing to StuTally! We welcome all types of contributions, whether it's fixing bugs, adding new features, improving documentation, or even reporting issues.

---

## 📋 Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [How Can I Contribute?](#how-can-i-contribute)
3. [Reporting Bugs](#reporting-bugs)
4. [Suggesting New Features](#suggesting-new-features)
5. [Contribution Workflow](#contribution-workflow)
6. [Coding Standards](#coding-standards)
7. [Writing Tests](#writing-tests)
8. [Code Documentation](#code-documentation)

---

## 🌟 Code of Conduct

### Our Commitment

We are committed to providing a welcoming and inclusive environment for all contributors. We expect everyone to:

- Use welcoming and inclusive language
- Respect different viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## 🎯 How Can I Contribute?

There are several ways to contribute to StuTally:

### 1. Reporting Bugs 🐛

If you find a bug:
- Check if it hasn't been reported in [Issues](https://github.com/aseelalmutari/StuTally-Project/issues)
- Open a new Issue with a detailed description
- Provide sufficient information to reproduce the problem

### 2. Fixing Bugs 🔧

- Browse [Issues](https://github.com/aseelalmutari/StuTally-Project/issues) to find bugs to fix
- Issues labeled with `good first issue` are suitable for beginners

### 3. Adding New Features ✨

- Check the [Roadmap](README.md#-roadmap)
- Suggest new features by opening an Issue
- Ensure the feature aligns with project goals

### 4. Improving Documentation 📚

- Fix spelling or grammatical errors
- Add new examples
- Clarify unclear parts
- Translate documentation to other languages

### 5. Code Reviews 👀

- Review open Pull Requests
- Test changes locally
- Provide constructive feedback

---

## 🐛 Reporting Bugs

### Before Reporting

1. **Search existing Issues**: Make sure the problem hasn't been reported
2. **Update to latest**: Ensure you're using the latest version
3. **Check documentation**: The solution might be documented

### How to Write a Good Bug Report

Use the following template:

```markdown
## Problem Description
[Clear and concise description of the problem]

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Do '...'
4. See error

## Expected Behavior
[What should have happened]

## Actual Behavior
[What actually happened]

## Screenshots
[If applicable]

## Environment
- Operating System: [e.g., macOS 13.0, Windows 11, Ubuntu 22.04]
- Python Version: [e.g., 3.10.5]
- StuTally Version: [e.g., 2.0]
- CUDA/GPU: [Yes/No, Version]

## Additional Information
[Any other information that might be useful]

## Logs
```
[Paste logs from logs/app.log]
```
```

---

## 💡 Suggesting New Features

### Before Suggesting

1. Check the [Roadmap](README.md#-roadmap)
2. Search Issues to ensure it hasn't been suggested before
3. Consider compatibility with project goals

### How to Write a Good Feature Proposal

```markdown
## Summary
[Brief description of proposed feature]

## Motivation
[Why is this feature useful?]

## Proposed Solution
[How should the feature work?]

## Alternatives Considered
[Are there alternative ways to achieve the same goal?]

## Additional Information
[Diagrams, examples, links, etc.]
```

---

## 🔄 Contribution Workflow

### 1. Fork the Project

```bash
# Click "Fork" button on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/StuTally.git
cd StuTally
```

### 2. Set Up Local Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
# Edit .env as needed
```

### 3. Create a New Branch

```bash
# Create branch from main
git checkout -b feature/your-feature-name

# Or for bug fix
git checkout -b fix/bug-description
```

**Branch Naming**:
- `feature/` - For new features
- `fix/` - For bug fixes
- `docs/` - For documentation updates
- `refactor/` - For refactoring
- `test/` - For adding/updating tests

### 4. Make Changes

```bash
# Make required changes
# ...

# Add modified files
git add .

# Commit changes with descriptive message
git commit -m "Add: feature description"
```

**Commit Message Format**:
```
<type>: <description>

[optional body]

[optional footer]
```

**Available Types**:
- `Add`: Add new feature
- `Fix`: Fix bug
- `Update`: Update existing feature
- `Refactor`: Code restructuring
- `Docs`: Update documentation
- `Test`: Add/update tests
- `Style`: Formatting changes
- `Perf`: Performance improvement
- `Chore`: Maintenance tasks

**Examples**:
```bash
git commit -m "Add: support for RTSP camera streaming"
git commit -m "Fix: video feed not loading on Safari browser"
git commit -m "Update: improve YOLO detection accuracy"
git commit -m "Docs: add API documentation for analytics endpoints"
```

### 5. Test Changes

```bash
# Run tests
python -m pytest tests/

# Run application to ensure it works
python run.py
```

### 6. Push to Repository

```bash
# Push branch to your fork
git push origin feature/your-feature-name
```

### 7. Open Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your branch
4. Fill in PR template with required information
5. Wait for review

**Pull Request Template**:
```markdown
## Description
[Brief description of changes]

## Type
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other: ___________

## Changes
- Change 1
- Change 2
- ...

## Testing
[How were changes tested?]

## Screenshots
[If applicable]

## Checklist
- [ ] Code follows project style
- [ ] Added comments, especially in complex areas
- [ ] Updated documentation if needed
- [ ] Didn't add new warnings
- [ ] Added tests to prove fix/feature works
- [ ] All new and existing tests pass
```

---

## 📝 Coding Standards

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some exceptions:

```python
# ✅ Good
def process_video_frame(frame, model_name, confidence_threshold=0.5):
    """
    Process a single video frame using YOLO detection.
    
    Args:
        frame: Input video frame
        model_name: Name of YOLO model to use
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        Annotated frame with detections
    """
    # Implementation
    pass

# ❌ Bad
def ProcessVideoFrame(Frame,ModelName,confidence_threshold=0.5):
    # No docstring
    pass
```

### File Organization

```python
# Import Order
# 1. Standard library
import os
import sys
from datetime import datetime

# 2. Third-party libraries
import cv2
import numpy as np
from flask import Flask, render_template

# 3. Local libraries
from app.services import video_service
from database import init_db
```

### Naming

- **Classes**: `PascalCase` - e.g., `VideoService`, `ModelLoader`
- **Functions/Methods**: `snake_case` - e.g., `process_video()`, `get_analytics()`
- **Constants**: `UPPER_SNAKE_CASE` - e.g., `MAX_FILE_SIZE`, `DEFAULT_MODEL`
- **Private**: `_leading_underscore` - e.g., `_internal_method()`

### Comments and Documentation

```python
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (tuple): First box coordinates (x1, y1, x2, y2)
        box2 (tuple): Second box coordinates (x1, y1, x2, y2)
        
    Returns:
        float: IoU value between 0 and 1
        
    Example:
        >>> box1 = (10, 10, 50, 50)
        >>> box2 = (30, 30, 70, 70)
        >>> iou = calculate_iou(box1, box2)
        >>> print(f"IoU: {iou:.2f}")
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Return implementation
    pass
```

### Error Handling

```python
# ✅ Good
try:
    result = process_video(video_path)
except FileNotFoundError as e:
    logger.error(f"Video file not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error processing video: {e}", exc_info=True)
    return None

# ❌ Bad
try:
    result = process_video(video_path)
except:
    pass
```

---

## 🧪 Writing Tests

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_models.py           # Model tests
├── test_services.py         # Service tests
├── test_api.py              # API tests
└── test_database.py         # Database tests
```

### Example Test

```python
import pytest
from app.services.video_service import VideoService

class TestVideoService:
    """Test cases for VideoService"""
    
    def test_video_loading(self, sample_video_path):
        """Test that video loads successfully"""
        service = VideoService()
        result = service.load_video(sample_video_path)
        
        assert result is not None
        assert result['status'] == 'success'
        assert 'video_id' in result
    
    def test_video_processing_with_yolo(self, sample_video_path):
        """Test video processing with YOLO model"""
        service = VideoService()
        result = service.process_video(
            video_path=sample_video_path,
            model_name='yolov8s.pt'
        )
        
        assert result['detections_count'] > 0
        assert result['fps'] > 0
    
    def test_invalid_video_path(self):
        """Test handling of invalid video path"""
        service = VideoService()
        
        with pytest.raises(FileNotFoundError):
            service.load_video('nonexistent_video.mp4')
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific file
pytest tests/test_services.py

# Run specific test
pytest tests/test_services.py::TestVideoService::test_video_loading

# With code coverage
pytest --cov=app tests/

# With HTML report
pytest --cov=app --cov-report=html tests/
```

---

## 📖 Code Documentation

### Docstrings

Use Google Style format:

```python
def get_analytics_data(video_id=None, time_range='all', filters=None):
    """
    Retrieve analytics data from database with optional filters.
    
    This function queries the detections database and aggregates
    statistics based on the provided parameters.
    
    Args:
        video_id (str, optional): Specific video ID to filter by.
            If None, includes all videos. Defaults to None.
        time_range (str, optional): Time range for data.
            Options: 'daily', 'weekly', 'monthly', 'all'.
            Defaults to 'all'.
        filters (dict, optional): Additional filters to apply.
            Keys can include 'stage', 'action', 'min_confidence'.
            Defaults to None.
    
    Returns:
        dict: Dictionary containing analytics data with keys:
            - 'total_students' (int): Total unique students detected
            - 'entries' (int): Number of entry events
            - 'exits' (int): Number of exit events
            - 'timestamps' (list): List of datetime strings
            - 'counts' (list): Corresponding counts
    
    Raises:
        ValueError: If time_range is not a valid option
        DatabaseError: If database connection fails
    
    Example:
        >>> data = get_analytics_data(
        ...     video_id='abc123',
        ...     time_range='daily',
        ...     filters={'stage': 'High School'}
        ... )
        >>> print(f"Total students: {data['total_students']}")
        Total students: 42
    
    Note:
        This function requires an active database connection.
        See `database.py` for connection setup.
    """
    # Implementation
    pass
```

### Comments in Code

```python
# ✅ Good - comment explains "why"
# Skip 3 frames to improve performance while maintaining tracking accuracy
if frame_count % 3 != 0:
    continue

# ✅ Good - comment to explain complex algorithm
# Use Kalman Filter to predict object position in next frame
predicted_position = kalman_filter.predict(current_position)

# ❌ Bad - comment explains "what" (clear from code)
# Increment counter
counter += 1
```

---

## 🎨 Frontend Standards

### HTML/CSS

```html
<!-- ✅ Good -->
<div class="video-container">
    <video id="videoFeed" class="video-stream" autoplay></video>
    <div class="video-controls">
        <button id="playBtn" class="btn btn-primary">Play</button>
    </div>
</div>

<!-- ❌ Bad -->
<div style="width:100%;height:auto">
    <video id="v1" autoplay></video>
    <button onclick="play()">Play</button>
</div>
```

### JavaScript

```javascript
// ✅ Good
class VideoPlayer {
    constructor(videoElement) {
        this.video = videoElement;
        this.isPlaying = false;
        this.init();
    }
    
    init() {
        this.attachEventListeners();
    }
    
    attachEventListeners() {
        this.video.addEventListener('play', () => {
            this.isPlaying = true;
            this.updateUI();
        });
    }
}

// ❌ Bad
function play() {
    var v = document.getElementById('v1');
    v.play();
}
```

---

## 🏷️ Versioning

We follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

Example: 2.1.3
```

- **MAJOR**: Breaking changes incompatible with previous versions
- **MINOR**: New features compatible with existing code
- **PATCH**: Bug fixes compatible with existing code

---

## 📦 Releasing a New Version

### Checklist (Admins Only)

1. Update version number in `version.py`
2. Update `CHANGELOG.md`
3. Run all tests
4. Commit changes
5. Create version tag
6. Push to repository with tags

```bash
# Update version
echo "__version__ = '2.1.0'" > app/version.py

# Commit
git add .
git commit -m "Release: version 2.1.0"

# Create Tag
git tag -a v2.1.0 -m "Version 2.1.0 - Feature updates"

# Push
git push origin main --tags
```

---

## ❓ Questions?

If you have any questions about contributing:

- 💬 Open [Discussion](https://github.com/aseelalmutari/StuTally-Project/discussions)
- 📧 Email us at: caa73061@gmail.com
- 📖 Check [Documentation](README.md)

---

## 🙏 Special Thanks

Thank you to all [contributors](https://github.com/aseelalmutari/StuTally-Project/graphs/contributors) who helped develop StuTally!

---

<div align="center">

**Made with ❤️ by the StuTally Community**

[⬆️ Back to Top](#-contributing-guide-to-stutally)

</div>
