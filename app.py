import streamlit as st
import json
import os
from datetime import datetime
import hashlib
from pages.auth import login_user, register_user, logout_user
from pages.home import show_home
from pages.detection import show_detection
from pages.history import show_history

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Penyakit Daun Kentang",
    page_icon="🥔",
    layout="wide"
)

 # Tambahkan CSS untuk menyembunyikan struktur file
st.markdown("""
<style>
/* Sembunyikan struktur file di sidebar */
.css-1d391kg, .css-1rs6os, .css-17eq0hr, .css-1lcbmhc.e1fqkh3o0 {
    display: none !important;
}

/* Sembunyikan navigasi default streamlit */
[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Sembunyikan elemen file browser */
.css-pkbazv.e1fqkh3o4 {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)


# Inisialisasi session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'page' not in st.session_state:
    st.session_state.page = "login"

# Membuat folder yang diperlukan
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

def main():
    # Sidebar untuk navigasi
    if st.session_state.logged_in:
        with st.sidebar:
            st.write(f"👋 Selamat datang, **{st.session_state.username}**!")
            st.markdown("---")
            
            # Menu navigasi
            if st.button("🏠 Home", use_container_width=True):
                st.session_state.page = "home"
            if st.button("🔍 Deteksi", use_container_width=True):
                st.session_state.page = "detection"
            if st.button("📋 Riwayat", use_container_width=True):
                st.session_state.page = "history"
            
            st.markdown("---")
            if st.button("🚪 Logout", use_container_width=True):
                logout_user()
                st.rerun()

    # Menampilkan halaman berdasarkan status login dan navigasi
    if not st.session_state.logged_in:
        show_auth_page()
    else:
        if st.session_state.page == "home":
            show_home()
        elif st.session_state.page == "detection":
            show_detection()
        elif st.session_state.page == "history":
            show_history()

def show_auth_page():
    st.title("🥔 Aplikasi Deteksi Penyakit Daun Kentang")
    
    # Tab untuk Login dan Register
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", use_container_width=True)
            
            if submit_login:
                if login_user(username, password):
                    st.success("Login berhasil!")
                    st.session_state.page = "home"
                    st.rerun()
                else:
                    st.error("Username atau password salah!")
    
    with tab2:
        st.subheader("Register")
        with st.form("register_form"):
            new_username = st.text_input("Username Baru")
            new_password = st.text_input("Password Baru", type="password")
            confirm_password = st.text_input("Konfirmasi Password", type="password")
            submit_register = st.form_submit_button("Register", use_container_width=True)
            
            if submit_register:
                if not new_username:
                    st.error("Username tidak boleh kosong!")
                elif not new_password:
                    st.error("Password tidak boleh kosong!")
                elif new_password != confirm_password:
                    st.error("Password tidak cocok!")
                elif len(new_password) < 6:
                    st.error("Password minimal 6 karakter!")
                elif register_user(new_username, new_password):
                    st.success("Registrasi berhasil! Silakan login.")
                else:
                    st.error("Username sudah digunakan!")

if __name__ == "__main__":
    main()