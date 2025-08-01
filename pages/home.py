import streamlit as st
from pages.auth import get_user_info
from pages.history import load_history
import json

def show_home():
    st.title("🏠 Dashboard")
    st.markdown("---")
    
    # Informasi pengguna
    user_info = get_user_info(st.session_state.username)
    
    # Statistik pengguna
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="👤 Username",
            value=st.session_state.username
        )
    
    with col2:
        history = load_history()
        user_detections = [h for h in history if h['username'] == st.session_state.username]
        st.metric(
            label="🔍 Total Deteksi",
            value=len(user_detections)
        )
    
    with col3:
        if user_detections:
            healthy_count = len([h for h in user_detections if h['prediction'] == 'Healthy'])
            disease_count = len(user_detections) - healthy_count
            st.metric(
                label="🌿 Healthy Leaf",
                value=healthy_count,
                delta=f"Penyakit: {disease_count}"
            )
        else:
            st.metric(
                label="🌿 Healthy Leaf",
                value=0
            )
    
    st.markdown("---")
    
    # Informasi tentang aplikasi
    st.subheader("📋 Tentang Aplikasi")
    st.write("""
    Aplikasi ini menggunakan CNN dengan Aristektur EfficientNetB3 untuk mendeteksi penyakit pada daun kentang.
    Aplikasi dapat mendeteksi kondisi berikut:
    
    - **Healthy**       : Daun kentang sehat
    - **Early Blight**  : Infeksi yang disebabkan oleh jamur Alternaria solani
    - **Late Blight**   : Penyakit serius yang disebabkan oleh Phytophthora infestans
    
    ### Cara Menggunakan:
    1. Klik menu **Deteksi** di sidebar
    2. Upload gambar daun kentang
    3. Tunggu hasil prediksi
    4. Lihat riwayat deteksi di menu **Riwayat**
    """)
    
    # Deteksi terbaru
    if user_detections:
        st.subheader("🕒 Deteksi Terbaru")
        latest_detection = sorted(user_detections, key=lambda x: x['timestamp'], reverse=True)[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if latest_detection.get('image_path'):
                try:
                    st.image(latest_detection['image_path'], caption="Gambar Terakhir", width=200)
                except:
                    st.write("Gambar tidak tersedia")
        
        with col2:
            st.write(f"**Hasil Prediksi:** {latest_detection['prediction']}")
            st.write(f"**Confidence:** {latest_detection['confidence']:.2f}%")
            st.write(f"**Waktu:** {latest_detection['timestamp']}")
    
    st.markdown("---")
    st.info("💡 Gunakan menu di sidebar untuk navigasi ke fitur lainnya!")