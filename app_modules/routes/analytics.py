# app/routes/analytics.py
"""
Analytics routes blueprint.
Handles all analytics, KPIs, and reporting endpoints.
"""
from flask import Blueprint, render_template, request, jsonify, send_file, Response
from flask_login import login_required
import logging

logger = logging.getLogger(__name__)

analytics_bp = Blueprint('analytics', __name__, url_prefix='/analytics')


@analytics_bp.route('/')
@login_required
def analytics_dashboard():
    """
    Main analytics dashboard page.
    Requires authentication.
    """
    return render_template('analytics.html')


@analytics_bp.route('/data')
@login_required
def analytics_data():
    """
    Get analytics data based on filters.
    
    Query params:
        - video_id: Filter by video
        - time_range: Time range filter
        - date: Specific date
        - stat_type: Type of statistic
    """
    from app.services.analytics_service import get_filtered_analytics
    
    try:
        video_id = request.args.get('video_id')
        time_range = request.args.get('time_range', 'all_time')
        date_str = request.args.get('date')
        stat_type = request.args.get('stat_type', 'total_students')
        
        data = get_filtered_analytics(
            video_id=video_id,
            time_range=time_range,
            date=date_str,
            stat_type=stat_type
        )
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error fetching analytics data: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@analytics_bp.route('/kpis')
@login_required
def analytics_kpis():
    """Get Key Performance Indicators"""
    from database import get_kpis
    
    try:
        video_id = request.args.get('video_id')
        data = get_kpis(video_id=video_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching KPIs: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/students_over_time')
@login_required
def analytics_students_over_time():
    """Get students count over time"""
    from database import get_students_over_time
    
    try:
        video_id = request.args.get('video_id')
        data = get_students_over_time(video_id=video_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching time series: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/heatmap_data')
@login_required
def analytics_heatmap_data():
    """Get heatmap data for peak times"""
    from database import get_heatmap_data
    
    try:
        video_id = request.args.get('video_id')
        data = get_heatmap_data(video_id=video_id)
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching heatmap: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/stage_comparison_data')
@login_required
def analytics_stage_comparison_data():
    """Get stage comparison data"""
    from database import get_stage_comparison_data
    
    try:
        data = get_stage_comparison_data()
        return jsonify(data)
    except Exception as e:
        logger.error(f"Error fetching stage comparison: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/download')
@login_required
def analytics_download():
    """Download analytics as CSV"""
    from app.services.export_service import export_csv
    
    try:
        video_id = request.args.get('video_id')
        stat_type = request.args.get('stat_type', 'total_students')
        
        csv_data = export_csv(video_id=video_id, stat_type=stat_type)
        
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=analytics.csv"}
        )
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@analytics_bp.route('/download_pdf')
@login_required
def analytics_download_pdf():
    """Download analytics as PDF"""
    from app.services.export_service import export_pdf
    
    try:
        video_id = request.args.get('video_id')
        stat_type = request.args.get('stat_type', 'total_students')
        
        pdf_buffer = export_pdf(video_id=video_id, stat_type=stat_type)
        
        return send_file(
            pdf_buffer,
            as_attachment=True,
            download_name='analytics.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

