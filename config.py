import os
from datetime import timedelta

class Config:
    """Base configuration"""
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    
    # Database
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Gemini API
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') or "AIzaSyB_dNcoAstUzFWW3c_N5vHtl08YfSYSxOA"
    DEFAULT_MODEL = "gemini-2.5-flash-preview-04-17"
    
    # File storage
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB max upload
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(days=7)
    
    # Celery configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    
    # Flask-Login configuration
    REMEMBER_COOKIE_DURATION = timedelta(days=14)
    
    # API rate limiting
    API_RATE_LIMIT = "100 per day"
    
    # Feature flags
    ENABLE_USER_REGISTRATION = True
    ENABLE_API_ACCESS = True
    ENABLE_TEMPLATE_SHARING = True
    ENABLE_DATA_VISUALIZATION = True
    ENABLE_MULTI_MODEL_SUPPORT = True
    ENABLE_DARK_MODE = True
    
    @staticmethod
    def init_app(app):
        """Initialize application with this configuration"""
        # Create upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'dev.db')


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or 'sqlite://'
    WTF_CSRF_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(os.path.dirname(os.path.abspath(__file__)), 'instance', 'prod.db')
    
    @classmethod
    def init_app(cls, app):
        """Initialize production application"""
        Config.init_app(app)
        
        # Log to stderr
        import logging
        from logging import StreamHandler
        file_handler = StreamHandler()
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)


# Map config name to config class
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
