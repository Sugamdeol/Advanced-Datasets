import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from celery import Celery

# Initialize SQLAlchemy
db = SQLAlchemy()

# Initialize LoginManager
login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.login_message_category = 'info'

# Initialize Celery
celery = Celery(__name__)

def create_app(config_name='default'):
    """Application factory"""
    app = Flask(__name__)
    
    # Load configuration
    from config import config
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Initialize extensions
    db.init_app(app)
    login_manager.init_app(app)
    
    # Configure Celery
    celery.conf.update(app.config)
    
    # Configure Google GenerativeAI
    import google.generativeai as genai
    genai.configure(api_key=app.config['GEMINI_API_KEY'])
    
    # Register blueprints
    from app.routes.main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    from app.routes.auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')
    
    from app.routes.datasets import datasets as datasets_blueprint
    app.register_blueprint(datasets_blueprint, url_prefix='/datasets')
    
    from app.routes.templates import templates as templates_blueprint
    app.register_blueprint(templates_blueprint, url_prefix='/templates')
    
    from app.routes.profile import profile as profile_blueprint
    app.register_blueprint(profile_blueprint, url_prefix='/profile')
    
    from app.api.v1 import api as api_v1_blueprint
    app.register_blueprint(api_v1_blueprint, url_prefix='/api/v1')
    
    # Load user from database for session
    from app.models.user import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Register custom error handlers
    register_error_handlers(app)
    
    # Register context processors
    register_context_processors(app)
    
    return app

def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(404)
    def page_not_found(e):
        from flask import render_template
        return render_template('errors/404.html'), 404
    
    @app.errorhandler(500)
    def internal_server_error(e):
        from flask import render_template
        return render_template('errors/500.html'), 500

def register_context_processors(app):
    """Register context processors"""
    
    @app.context_processor
    def utility_processor():
        """Add utility functions to template context"""
        def format_date(date, format='%Y-%m-%d %H:%M'):
            if date:
                return date.strftime(format)
            return ''
            
        def get_model_options():
            return [
                {'id': 'gemini-2.5-flash-preview-04-17', 'name': 'Gemini 2.5 Flash Preview'},
                {'id': 'gemini-2.5-pro', 'name': 'Gemini 2.5 Pro'},
                {'id': 'gemini-1.5-pro', 'name': 'Gemini 1.5 Pro'},
                {'id': 'gemini-1.5-flash', 'name': 'Gemini 1.5 Flash'}
            ]
        
        return dict(
            format_date=format_date,
            get_model_options=get_model_options
        )
