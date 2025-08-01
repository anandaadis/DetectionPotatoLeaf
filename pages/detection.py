import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json

def load_model():
    """Load model keras"""
    try:
        model = tf.keras.models.load_model('models/model_potato.keras')
        class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("Pastikan file 'model_potato.keras' ada di folder 'models/'")
        return None

def preprocess_image(image):
    """Preprocess gambar untuk prediksi - SESUAI TRAINING"""
    try:
        # Pastikan gambar dalam mode RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ke ukuran SAMA dengan training: (300, 300)
        img = image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Convert ke numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Pastikan shape yang benar (300, 300, 3)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 1:  # Single channel
            img_array = np.repeat(img_array, 3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Ambil RGB saja
        
        # Pastikan shape (300, 300, 3)
        assert img_array.shape == (300, 300, 3), f"Expected shape (300, 300, 3), got {img_array.shape}"
        
        # Normalisasi ke range [0, 1] seperti EfficientNet preprocessing
        img_array = img_array / 255.0
        
        # Tambahkan batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return None


def predict_disease(model, image):
    """Prediksi penyakit dari gambar"""
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None, None
            
        predictions = model.predict(processed_image)
        
        # Class names - sesuaikan dengan class yang digunakan saat training
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)

        # Bersihkan nama class untuk display
        clean_prediction = class_names[predicted_class].replace('Potato___', '').replace('_', ' ')
        
        return class_names[predicted_class], confidence, predictions[0]
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

def save_detection_history(username, prediction, confidence, image_path):
    """Simpan riwayat deteksi"""
    try:
        history_file = 'data/history.json'
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        # Load existing history
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        # Add new detection
        new_detection = {
            'username': username,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path
        }
        
        history.append(new_detection)
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
            
    except Exception as e:
        st.warning(f"Could not save history: {str(e)}")

def show_detection():
    st.title("🔍 Deteksi Penyakit Daun Kentang")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()

    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    
    st.write("Upload gambar daun kentang untuk mendeteksi penyakit:")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload gambar daun kentang dalam format PNG, JPG, atau JPEG"
    )
    
    if uploaded_file is not None:
        # Tampilkan gambar yang diupload
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Gambar Input")
                st.image(image, caption="Gambar yang diupload", use_column_width=True)
            
            with col2:
                st.subheader("Hasil Deteksi")
                
                # Tombol untuk prediksi
                if st.button("🔍 Analisis Gambar", use_container_width=True):
                    with st.spinner("Menganalisis gambar..."):
                        # Prediksi
                        prediction, confidence, all_predictions = predict_disease(model, image)
                        
                        if prediction is not None:
                            formatted_prediction = prediction.replace("Potato___", "").replace("_", " ").title()
                            
                            # Simpan gambar (dengan error handling untuk deployment)
                            try:
                                os.makedirs("uploads", exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                image_filename = f"{st.session_state.get('username', 'user')}_{timestamp}.jpg"
                                image_path = f"uploads/{image_filename}"
                                image.save(image_path)
                            except Exception as e:
                                st.warning(f"Could not save image: {str(e)}")
                                image_path = "temp_image.jpg"
                            
                            # Tampilkan hasil dengan formatting yang benar
                            clean_pred = prediction.replace("Potato___", "").replace("_", " ").title()
                            
                            if "healthy" in prediction.lower():
                                st.success(f"🌿 **Hasil: {clean_pred}**")
                                st.balloons()
                            elif "early" in prediction.lower():
                                st.warning(f"⚠️ **Hasil: {clean_pred}**")
                            else:  # Late blight
                                st.error(f"🚨 **Hasil: {clean_pred}**")
                            
                            st.write(f"**Confidence Score:** {confidence:.2f}%")
                            
                            # Progress bar untuk confidence
                            st.progress(confidence / 100)
                            
                            # Tampilkan semua probabilitas
                            with st.expander("Detail Probabilitas"):
                                for class_name, prob in zip(class_names, all_predictions):
                                    label = class_name.replace("Potato___", "").replace("_", " ").title()
                                    st.write(f"**{label}:** {prob*100:.2f}%")
                            
                            # Simpan ke riwayat
                            username = st.session_state.get('username', 'anonymous')
                            save_detection_history(
                                username,
                                formatted_prediction,
                                confidence,
                                image_path
                            )
                        else:
                            st.error("❌ Gagal menganalisis gambar. Silakan coba lagi.")
                            
        except Exception as e:
            st.error(f"Error processing uploaded file: {str(e)}")
    
    st.markdown("---")
    st.info("💡 **Tips:** Untuk hasil terbaik, gunakan gambar dengan pencahayaan yang baik dan fokus pada daun yang ingin dianalisis.")
