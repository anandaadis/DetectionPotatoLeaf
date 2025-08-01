import streamlit as st
import json
import hashlib
import os
from datetime import datetime

def hash_password(password):
    """Hash password menggunakan SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load data pengguna dari file JSON"""
    if os.path.exists('data/users.json'):
        with open('data/users.json', 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Simpan data pengguna ke file JSON"""
    with open('data/users.json', 'w') as f:
        json.dump(users, f, indent=4)

def register_user(username, password):
    """Registrasi pengguna baru"""
    users = load_users()
    
    if username in users:
        return False
    
    users[username] = {
        'password': hash_password(password),
        'created_at': datetime.now().isoformat(),
        'total_detections': 0
    }
    
    save_users(users)
    return True

def login_user(username, password):
    """Login pengguna"""
    users = load_users()
    
    if username in users:
        if users[username]['password'] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            return True
    
    return False

def logout_user():
    """Logout pengguna"""
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.page = "login"

def get_user_info(username):
    """Mendapatkan informasi pengguna"""
    users = load_users()
    return users.get(username, {})