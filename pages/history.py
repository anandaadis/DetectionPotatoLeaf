import streamlit as st
import json
import os
import pandas as pd
from datetime import datetime
import plotly.express as px

def load_history():
    """Load riwayat deteksi dari file JSON"""
    if os.path.exists('data/history.json'):
        with open('data/history.json', 'r') as f:
            return json.load(f)
    return []

def format_timestamp(timestamp_str):
    """Format timestamp untuk tampilan yang lebih baik"""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime("%d/%m/%Y %H:%M")
    except:
        return timestamp_str

def show_history():
    st.title("📋 Riwayat Deteksi")
    st.markdown("---")
    
    # Load history
    all_history = load_history()
    user_history = [h for h in all_history if h['username'] == st.session_state.username]
    
    if not user_history:
        st.info("Belum ada riwayat deteksi. Mulai deteksi pertama Anda di menu Deteksi!")
        return
    
    # Statistik ringkasan
    st.subheader("📊 Ringkasan Statistik")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Deteksi", len(user_history))
    
    with col2:
        healthy_count = len([h for h in user_history if h['prediction'] == 'Healthy'])
        st.metric("Healthy Leaf", healthy_count)
    
    with col3:
        early_blight_count = len([h for h in user_history if h['prediction'] == 'Early Blight'])
        st.metric("Early Blight", early_blight_count)
    
    with col4:
        late_blight_count = len([h for h in user_history if h['prediction'] == 'Late Blight'])
        st.metric("Late Blight", late_blight_count)
    
    # Grafik distribusi hasil
    st.subheader("📈 Distribusi Hasil Deteksi")
    
    # Buat dataframe untuk visualisasi
    df = pd.DataFrame(user_history)
    
    if len(df) > 0:
        # Pie chart distribusi
        prediction_counts = df['prediction'].value_counts()
        fig_pie = px.pie(
            values=prediction_counts.values,
            names=prediction_counts.index,
            title="Distribusi Hasil Deteksi",
            color_discrete_map={
                'Healthy': '#28a745',
                'Early Blight': '#ffc107',
                'Late Blight': '#dc3545'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    
    # Filter dan pencarian
    st.subheader("🔍 Filter Riwayat")
    
    col1, col2 = st.columns(2)
    
    with col1:
        filter_prediction = st.selectbox(
            "Filter berdasarkan hasil:",
            ["Semua", "Healthy", "Early Blight", "Late Blight"]
        )
    
    with col2:
        sort_order = st.selectbox(
            "Urutkan berdasarkan:",
            ["Terbaru", "Terlama", "Confidence Tertinggi", "Confidence Terendah"]
        )
    
    # Apply filters
    filtered_history = user_history.copy()
    
    if filter_prediction != "Semua":
        filtered_history = [h for h in filtered_history if h['prediction'] == filter_prediction]
    
    # Apply sorting
    if sort_order == "Terbaru":
        filtered_history.sort(key=lambda x: x['timestamp'], reverse=True)
    elif sort_order == "Terlama":
        filtered_history.sort(key=lambda x: x['timestamp'])
    elif sort_order == "Confidence Tertinggi":
        filtered_history.sort(key=lambda x: x['confidence'], reverse=True)
    elif sort_order == "Confidence Terendah":
        filtered_history.sort(key=lambda x: x['confidence'])
    
    # Tampilkan riwayat
    st.subheader(f"📋 Daftar Riwayat ({len(filtered_history)} item)")
    
    if not filtered_history:
        st.info("Tidak ada data yang sesuai dengan filter.")
        return
    
    # Pagination
    items_per_page = 5
    total_pages = (len(filtered_history) - 1) // items_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Halaman:", range(1, total_pages + 1))
        start_idx = (page - 1) * items_per_page
        end_idx = start_idx + items_per_page
        page_history = filtered_history[start_idx:end_idx]
    else:
        page_history = filtered_history
    
    # Tampilkan setiap item riwayat
    for i, item in enumerate(page_history):
        with st.expander(f"Deteksi #{start_idx + i + 1 if 'start_idx' in locals() else i + 1} - {item['prediction']} ({format_timestamp(item['timestamp'])})"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if item.get('image_path') and os.path.exists(item['image_path']):
                    try:
                        st.image(item['image_path'], caption="Gambar", width=200)
                    except:
                        st.write("❌ Gambar tidak dapat ditampilkan")
                else:
                    st.write("❌ Gambar tidak tersedia")
            
            with col2:
                # Status badge
                if item['prediction'] == 'Healthy':
                    st.success(f"✅ **{item['prediction']}**")
                elif item['prediction'] == 'Early Blight':
                    st.warning(f"⚠️ **{item['prediction']}**")
                else:
                    st.error(f"🚨 **{item['prediction']}**")
                
                st.write(f"**Confidence:** {item['confidence']:.2f}%")
                st.write(f"**Waktu:** {format_timestamp(item['timestamp'])}")
                
                # Progress bar confidence
                st.progress(item['confidence'] / 100)
    
    # Export data
    if st.button("📥 Export Riwayat ke CSV"):
        df_export = pd.DataFrame(filtered_history)
        if len(df_export) > 0:
            df_export['formatted_timestamp'] = df_export['timestamp'].apply(format_timestamp)
            csv = df_export.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"riwayat_deteksi_{st.session_state.username}.csv",
                mime="text/csv"
            )