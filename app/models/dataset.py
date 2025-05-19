from datetime import datetime
import json
from app import db

class Dataset(db.Model):
    """Dataset model to store generated dataset metadata"""
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    prompt = db.Column(db.Text, nullable=False)
    row_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    file_path = db.Column(db.String(255))
    file_size = db.Column(db.Integer, default=0)  # Size in bytes
    model_used = db.Column(db.String(64))
    generation_time = db.Column(db.Float)  # Time in seconds
    format = db.Column(db.String(20), default='csv')  # csv, json, excel, etc.
    is_public = db.Column(db.Boolean, default=False)
    download_count = db.Column(db.Integer, default=0)
    rating = db.Column(db.Float, default=0.0)  # User rating
    
    # Foreign Keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    template_id = db.Column(db.Integer, db.ForeignKey('dataset_templates.id'), nullable=True)
    
    # Relationships
    columns = db.relationship('DatasetColumn', backref='dataset', lazy='dynamic', 
                             cascade='all, delete-orphan')
    
    def __init__(self, name, description, prompt, user_id, model_used, row_count=0):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.user_id = user_id
        self.model_used = model_used
        self.row_count = row_count
    
    def increment_download_count(self):
        """Increment download counter"""
        self.download_count += 1
    
    def add_column(self, name, data_type=None, constraints=None):
        """Add a column to the dataset"""
        column = DatasetColumn(name=name, data_type=data_type, 
                              constraints=json.dumps(constraints) if constraints else None,
                              dataset_id=self.id)
        db.session.add(column)
        return column
    
    def to_dict(self):
        """Return dataset data as dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'prompt': self.prompt,
            'row_count': self.row_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'model_used': self.model_used,
            'generation_time': self.generation_time,
            'format': self.format,
            'is_public': self.is_public,
            'download_count': self.download_count,
            'rating': self.rating,
            'columns': [column.to_dict() for column in self.columns]
        }
        
    def __repr__(self):
        return f'<Dataset {self.name}>'


class DatasetColumn(db.Model):
    """Model to store column information for datasets"""
    __tablename__ = 'dataset_columns'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    display_name = db.Column(db.String(64))
    description = db.Column(db.Text)
    data_type = db.Column(db.String(32))  # string, integer, float, date, etc.
    constraints = db.Column(db.Text)  # JSON string of constraints
    order = db.Column(db.Integer, default=0)
    is_required = db.Column(db.Boolean, default=True)
    
    # Foreign Keys
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'))
    template_column_id = db.Column(db.Integer, db.ForeignKey('dataset_template_columns.id'), nullable=True)
    
    def __init__(self, name, dataset_id, data_type=None, constraints=None, order=0):
        self.name = name
        self.dataset_id = dataset_id
        self.data_type = data_type
        self.constraints = constraints
        self.order = order
        self.display_name = name.replace('_', ' ').title()
    
    def get_constraints(self):
        """Get constraints as dictionary"""
        if self.constraints:
            return json.loads(self.constraints)
        return {}
    
    def to_dict(self):
        """Return column data as dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'data_type': self.data_type,
            'constraints': self.get_constraints(),
            'order': self.order,
            'is_required': self.is_required
        }
        
    def __repr__(self):
        return f'<DatasetColumn {self.name}>'


class DatasetTemplateColumn(db.Model):
    """Model to store column templates"""
    __tablename__ = 'dataset_template_columns'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    display_name = db.Column(db.String(64))
    description = db.Column(db.Text)
    data_type = db.Column(db.String(32))
    constraints = db.Column(db.Text)  # JSON string of constraints
    order = db.Column(db.Integer, default=0)
    is_required = db.Column(db.Boolean, default=True)
    example_values = db.Column(db.Text)  # JSON array of example values
    
    # Foreign Keys
    template_id = db.Column(db.Integer, db.ForeignKey('dataset_templates.id'))
    
    def to_dict(self):
        """Return column template data as dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'data_type': self.data_type,
            'constraints': json.loads(self.constraints) if self.constraints else {},
            'order': self.order,
            'is_required': self.is_required,
            'example_values': json.loads(self.example_values) if self.example_values else []
        }


class DatasetTemplate(db.Model):
    """Model to store dataset templates"""
    __tablename__ = 'dataset_templates'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    category = db.Column(db.String(64))
    prompt_template = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public = db.Column(db.Boolean, default=False)
    usage_count = db.Column(db.Integer, default=0)
    
    # Foreign Keys
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    columns = db.relationship('DatasetTemplateColumn', backref='template', lazy='dynamic',
                             cascade='all, delete-orphan')
    datasets = db.relationship('Dataset', backref='template', lazy='dynamic')
    
    def increment_usage(self):
        """Increment usage counter"""
        self.usage_count += 1
    
    def to_dict(self):
        """Return template data as dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'prompt_template': self.prompt_template,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_public': self.is_public,
            'usage_count': self.usage_count,
            'columns': [column.to_dict() for column in self.columns]
        }
        
    def __repr__(self):
        return f'<DatasetTemplate {self.name}>'
