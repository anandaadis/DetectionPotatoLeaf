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
        # Cek beberapa kemungkinan lokasi model
        possible_paths = [
            'models/model_potato.keras',
            'model_potato.keras',
            'models/model_potato.h5',
            'model_potato.h5'
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            st.error("❌ Model file tidak ditemukan!")
            st.error("Pastikan salah satu file berikut ada:")
            for path in possible_paths:
                st.error(f"  - {path}")
            return None
            
        model = tf.keras.models.load_model(model_path)
        st.success(f"✅ Model berhasil dimuat dari: {model_path}")
        
        # Debug: tampilkan informasi model
        st.info(f"Model input shape: {model.input_shape}")
        st.info(f"Model output shape: {model.output_shape}")
        
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Pastikan model kompatibel dengan versi TensorFlow yang digunakan")
        return None

def preprocess_image(image, target_size=(300, 300)):
    """
    Preprocess gambar untuk prediksi - SESUAI DENGAN TRAINING ORIGINAL
    Kembalikan ke ukuran 300x300 seperti training asli
    """
    try:
        # Convert ke RGB jika perlu
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ke ukuran 300x300 (sesuai training original)
        img = image.resize(target_size)
        img_array = np.array(img)
        
        # Pastikan format channel RGB (3 channel)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Ambil hanya RGB
        
        # Preprocessing untuk EfficientNet - normalisasi ke [-1, 1]
        # EfficientNet biasanya menggunakan preprocessing khusus
        img_array = img_array.astype(np.float32)
        # Normalisasi ImageNet standard untuk EfficientNet
        img_array = (img_array / 127.5) - 1.0  # Scale ke [-1, 1]
        
        # Tambahkan batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_disease(model, image):
    """Prediksi penyakit dari gambar"""
    try:
        # Gunakan ukuran 300x300 sesuai training original
        processed_image = preprocess_image(image, target_size=(300, 300))
        if processed_image is None:
            return None, None, None
            
        predictions = model.predict(processed_image, verbose=0)
        
        # Class names - PASTIKAN URUTAN SESUAI TRAINING
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        # Clean prediction name
        prediction_label = class_names[predicted_class].replace('Potato___', '').replace('_', ' ')
        
        return prediction_label, confidence, predictions[0]
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        # Tampilkan detail error untuk debugging
        st.error(f"Error details: {type(e).__name__}: {str(e)}")
        return None, None, None

def save_detection_history(username, prediction, confidence, image_path):
    """Simpan riwayat deteksi"""
    try:
        # Pastikan folder data ada
        os.makedirs('data', exist_ok=True)
        
        history_file = 'data/history.json'
        
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
        st.warning(f"Tidak dapat menyimpan riwayat: {str(e)}")

def show_detection():
    st.title("🔍 Deteksi Penyakit Daun Kentang")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ Aplikasi tidak dapat berjalan tanpa model yang valid")
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
        except Exception as e:
            st.error(f"Error membuka gambar: {str(e)}")
            return
        
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
                        # Pastikan folder uploads ada
                        os.makedirs('uploads', exist_ok=True)
                        
                        # Simpan gambar
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        username = st.session_state.get('username', 'anonymous')
                        image_filename = f"{username}_{timestamp}.jpg"
                        image_path = f"uploads/{image_filename}"
                        
                        try:
                            image.save(image_path)
                        except Exception as e:
                            st.warning(f"Tidak dapat menyimpan gambar: {str(e)}")
                            image_path = "temp_image"
                        
                        # Tampilkan hasil
                        if "healthy" in prediction.lower():
                            st.success(f"🌿 **Hasil: {prediction.title()}**")
                            st.balloons()
                        elif "early" in prediction.lower():
                            st.warning(f"⚠️ **Hasil: {prediction.title()}**")
                        else:  # Late blight
                            st.error(f"🚨 **Hasil: {prediction.title()}**")
                        
                        st.write(f"**Confidence Score:** {confidence:.2f}%")
                        
                        # Progress bar untuk confidence
                        st.progress(confidence / 100)
                        
                        # Tampilkan semua probabilitas
                        with st.expander("Detail Probabilitas"):
                            for class_name, prob in zip(class_names, all_predictions):
                                clean_name = class_name.replace("Potato___", "").replace("_", " ").title()
                                st.write(f"**{clean_name}:** {prob*100:.2f}%")
                        
                        # Simpan ke riwayat jika username tersedia
                        if 'username' in st.session_state:
                            save_detection_history(
                                st.session_state.username,
                                prediction,
                                confidence,
                                image_path
                            )
                    else:
                        st.error("❌ Gagal melakukan prediksi. Silakan coba gambar lain atau periksa model.")
    
    st.markdown("---")
    st.info("💡 **Tips:** Untuk hasil terbaik, gunakan gambar dengan pencahayaan yang baik dan fokus pada daun yang ingin dianalisis.")

# Tambahan untuk debugging di Streamlit Cloud
def debug_environment():
    """Debug environment information"""
    with st.expander("🔧 Debug Information"):
        st.write("**TensorFlow Version:**", tf.__version__)
        st.write("**Current Working Directory:**", os.getcwd())
        st.write("**Files in current directory:**")
        for item in os.listdir('.'):
            st.write(f"  - {item}")
        
        if os.path.exists('models'):
            st.write("**Files in models directory:**")
            for item in os.listdir('models'):
                st.write(f"  - {item}")
        else:
            st.write("**models directory does not exist**")

# Uncomment baris di bawah untuk debugging
# debug_environment()
