# app/services/export_service.py
"""
Export service.
Handles exporting analytics data to different formats (CSV, PDF, Excel).
"""
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import logging

logger = logging.getLogger(__name__)


def export_csv(video_id=None, stat_type='total_students'):
    """
    Export analytics data as CSV.
    
    Args:
        video_id: Optional video filter
        stat_type: Type of statistic
        
    Returns:
        CSV string
    """
    from database import get_analytics
    
    try:
        data = get_analytics(video_id=video_id)
        df = pd.DataFrame(data, columns=['class_name', 'stage', 'count'])
        csv_data = df.to_csv(index=False)
        
        logger.info(f"CSV export generated for video_id={video_id}")
        return csv_data
        
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}", exc_info=True)
        raise


def export_pdf(video_id=None, stat_type='total_students'):
    """
    Export analytics data as PDF.
    
    Args:
        video_id: Optional video filter
        stat_type: Type of statistic
        
    Returns:
        BytesIO buffer with PDF data
    """
    from database import get_analytics
    
    try:
        data = get_analytics(video_id=video_id)
        df = pd.DataFrame(data, columns=['class_name', 'stage', 'count'])
        
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, "StuTally Analytics Report")
        
        # Data
        c.setFont("Helvetica", 10)
        y = height - 100
        
        for index, row in df.iterrows():
            text = f"Category: {row['class_name']} - Stage: {row['stage']} - Count: {row['count']}"
            c.drawString(50, y, text)
            y -= 20
            
            if y < 50:
                c.showPage()
                c.setFont("Helvetica", 10)
                y = height - 50
        
        c.save()
        buffer.seek(0)
        
        logger.info(f"PDF export generated for video_id={video_id}")
        return buffer
        
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}", exc_info=True)
        raise


def export_excel(video_id=None, stat_type='total_students'):
    """
    Export analytics data as Excel.
    (Placeholder for future implementation)
    
    Args:
        video_id: Optional video filter
        stat_type: Type of statistic
        
    Returns:
        BytesIO buffer with Excel data
    """
    # TODO: Implement Excel export with openpyxl or xlsxwriter
    raise NotImplementedError("Excel export not yet implemented")

