from flask import Blueprint, render_template, redirect, url_for, flash, request
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models.user import User
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            flash('Please check your login details and try again.', 'danger')
            return redirect(url_for('auth.login'))
        
        # Update last login time
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        # Log in user
        login_user(user, remember=remember)
        
        # Redirect to the page the user was trying to access
        next_page = request.args.get('next')
        if next_page and next_page.startswith('/'):
            return redirect(next_page)
        return redirect(url_for('main.index'))
    
    return render_template('auth/login.html')

@auth.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Check if passwords match
        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('auth.register'))
        
        # Check if email already exists
        user_email = User.query.filter_by(email=email).first()
        if user_email:
            flash('Email address already exists.', 'danger')
            return redirect(url_for('auth.register'))
        
        # Check if username already exists
        user_name = User.query.filter_by(username=username).first()
        if user_name:
            flash('Username already exists.', 'danger')
            return redirect(url_for('auth.register'))
        
        # Create new user
        new_user = User(
            username=username,
            email=email,
            password=password
        )
        
        # Generate API key for the user
        new_user.generate_api_key()
        
        # Add user to database
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! You can now log in.', 'success')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/register.html')

@auth.route('/logout')
@login_required
def logout():
    """Handle user logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index'))

@auth.route('/reset-password', methods=['GET', 'POST'])
def reset_password_request():
    """Handle password reset request"""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            # In a real application, you would send an email with a token
            # For this demo, we'll just flash a message
            flash('Password reset instructions have been sent to your email.', 'info')
        else:
            flash('Email not found.', 'danger')
        
        return redirect(url_for('auth.login'))
    
    return render_template('auth/reset_password_request.html')

@auth.route('/profile')
@login_required
def profile():
    """View user profile"""
    return redirect(url_for('profile.view'))
