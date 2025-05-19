from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from app import db

class User(db.Model, UserMixin):
    """User model for authentication and tracking dataset generation"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, index=True, nullable=False)
    email = db.Column(db.String(120), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_admin = db.Column(db.Boolean, default=False)
    theme_preference = db.Column(db.String(20), default='light')
    api_key = db.Column(db.String(64), unique=True, index=True)
    api_requests_count = db.Column(db.Integer, default=0)
    
    # Relationships
    datasets = db.relationship('Dataset', backref='creator', lazy='dynamic')
    templates = db.relationship('DatasetTemplate', backref='creator', lazy='dynamic')
    
    def __init__(self, username, email, password, is_admin=False):
        self.username = username
        self.email = email
        self.set_password(password)
        self.is_admin = is_admin
    
    def set_password(self, password):
        """Set password hash"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password against hash"""
        return check_password_hash(self.password_hash, password)
    
    def generate_api_key(self):
        """Generate a unique API key for the user"""
        import secrets
        self.api_key = secrets.token_hex(32)
        return self.api_key
    
    def increment_api_requests(self):
        """Increment API request counter"""
        self.api_requests_count += 1
        
    def to_dict(self):
        """Return user data as dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'theme_preference': self.theme_preference,
            'datasets_count': self.datasets.count()
        }
    
    def __repr__(self):
        return f'<User {self.username}>'
