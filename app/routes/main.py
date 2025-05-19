from flask import Blueprint, render_template, redirect, url_for, flash, current_app
from flask_login import login_required, current_user
from app.models.dataset import Dataset, DatasetTemplate
from app import db

main = Blueprint('main', __name__)

@main.route('/')
def index():
    """Render the homepage"""
    # Get public templates for showcase
    public_templates = DatasetTemplate.query.filter_by(is_public=True).order_by(DatasetTemplate.usage_count.desc()).limit(3).all()
    
    # Get total datasets generated and users count
    stats = {
        'datasets_count': Dataset.query.count(),
        'users_count': db.session.query(db.func.count(db.distinct(Dataset.user_id))).scalar() or 0,
        'templates_count': DatasetTemplate.query.count()
    }
    
    # Get featured datasets if user is logged in
    featured_datasets = []
    if current_user.is_authenticated:
        # Get user's recent datasets
        user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).limit(3).all()
        featured_datasets = user_datasets
    else:
        # Get some public datasets
        featured_datasets = Dataset.query.filter_by(is_public=True).order_by(Dataset.download_count.desc()).limit(3).all()
    
    return render_template('main/index.html', 
                          public_templates=public_templates,
                          featured_datasets=featured_datasets,
                          stats=stats)

@main.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    # Get user's datasets
    user_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).all()
    
    # Get user's templates
    user_templates = DatasetTemplate.query.filter_by(user_id=current_user.id).order_by(DatasetTemplate.updated_at.desc()).all()
    
    # Get user stats
    stats = {
        'datasets_count': len(user_datasets),
        'templates_count': len(user_templates),
        'downloads_count': sum(dataset.download_count for dataset in user_datasets),
        'api_requests': current_user.api_requests_count
    }
    
    return render_template('main/dashboard.html',
                          user_datasets=user_datasets,
                          user_templates=user_templates, 
                          stats=stats)

@main.route('/about')
def about():
    """About page"""
    return render_template('main/about.html')

@main.route('/contact')
def contact():
    """Contact page"""
    return render_template('main/contact.html')

@main.route('/docs')
def docs():
    """Documentation page"""
    return render_template('main/docs.html')

@main.route('/api-docs')
def api_docs():
    """API Documentation page"""
    return render_template('main/api_docs.html')

@main.route('/theme/<theme>')
def set_theme(theme):
    """Set user theme preference"""
    valid_themes = ['light', 'dark', 'auto']
    
    if theme not in valid_themes:
        flash('Invalid theme selection.', 'warning')
        return redirect(url_for('main.index'))
    
    if current_user.is_authenticated:
        current_user.theme_preference = theme
        db.session.commit()
        flash(f'Theme set to {theme.capitalize()} mode.', 'success')
    else:
        from flask import session
        session['theme'] = theme
        flash(f'Theme set to {theme.capitalize()} mode for this session.', 'success')
    
    # Redirect back to the referring page
    from flask import request
    next_page = request.referrer or url_for('main.index')
    return redirect(next_page)
