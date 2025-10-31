# auth.py
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token, jwt_required
import sqlite3

auth_bp = Blueprint('auth', __name__)
bcrypt = Bcrypt()
login_manager = LoginManager()
jwt = JWTManager()

# Simple User class
class User(UserMixin):
    def __init__(self, id, username, role):
        self.id = id
        self.username = username
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('data/detections.db')
    row = conn.execute(
        "SELECT id, username, role FROM users WHERE id=?",
        (user_id,)
    ).fetchone()
    conn.close()
    return User(*row) if row else None

@auth_bp.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        u = request.form['username']
        p = request.form['password']
        conn = sqlite3.connect('data/detections.db')
        row = conn.execute(
            "SELECT id,password,role FROM users WHERE username=?",
            (u,)
        ).fetchone()
        conn.close()
        if row and bcrypt.check_password_hash(row[1], p):
            user = User(row[0], u, row[2])
            login_user(user)
            return redirect(request.args.get('next') or url_for('analytics'))
        flash('Invalid credentials', 'danger')
    return render_template('login.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))

# JWT login for API
@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    u,p = data.get('username'), data.get('password')
    conn = sqlite3.connect('data/detections.db')
    row = conn.execute(
      "SELECT password,role FROM users WHERE username=?",
      (u,)
    ).fetchone()
    conn.close()
    if row and bcrypt.check_password_hash(row[0], p):
        token = create_access_token(identity=u, additional_claims={'role':row[1]})
        return jsonify(access_token=token)
    return jsonify(msg="Bad credentials"), 401

# Protect API route
@auth_bp.route('/api/analytics')
@jwt_required()
def api_analytics():
    from database import get_latest_analytics
    return jsonify(get_latest_analytics())
