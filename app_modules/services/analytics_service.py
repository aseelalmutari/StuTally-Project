# app/services/analytics_service.py
"""
Analytics service.
Business logic for analytics data processing and filtering.
"""
import logging
from typing import Optional, Dict, Any
from flask import current_app

logger = logging.getLogger(__name__)


def get_filtered_analytics(
    video_id: Optional[str] = None,
    time_range: str = 'all_time',
    date: Optional[str] = None,
    stat_type: str = 'total_students'
) -> Dict[str, Any]:
    """
    Get analytics data with filters applied.
    
    Args:
        video_id: Filter by video ID
        time_range: Time range filter
        date: Specific date filter
        stat_type: Type of statistic
        
    Returns:
        Filtered analytics data
    """
    from database import (
        get_analytics,
        get_time_based_analytics,
        get_date_based_analytics
    )
    
    try:
        # Check if custom model is being used
        if stat_type == 'academic_stages':
            # Only available with custom model
            # Could add validation here
            data = get_analytics(video_id=video_id)
        elif time_range == 'all_time':
            data = get_analytics(video_id=video_id)
        elif time_range in ['daily', 'weekly', 'monthly', 'morning', 'afternoon']:
            data = get_time_based_analytics(video_id=video_id, time_range=time_range)
        elif time_range == 'date_based' and date:
            data = get_date_based_analytics(video_id=video_id, start_date=date, end_date=date)
        else:
            data = get_analytics(video_id=video_id)
        
        return data
        
    except Exception as e:
        logger.error(f"Error getting filtered analytics: {e}", exc_info=True)
        raise

