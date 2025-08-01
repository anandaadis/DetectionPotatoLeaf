import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json

def rebuild_model_wrapper():
    """Buat wrapper model yang kompatibel dengan model yang rusak"""
    try:
        # Load model yang bermasalah
        original_model = tf.keras.models.load_model('models/model_potato.keras')
        
        # Buat model wrapper baru dengan input yang benar
        inputs = tf.keras.Input(shape=(300, 300, 3))  # RGB input yang benar
        
        # Preprocessing layer untuk mengkonversi RGB ke grayscale jika diperlukan
        # Karena model asli sepertinya expect grayscale
        x = tf.keras.layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))(inputs)
        
        # Resize dari 300x300 ke 301x301 jika diperlukan
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [301, 301]))(x)
        
        # Gunakan layers dari model asli (skip input layer)
        for i, layer in enumerate(original_model.layers[1:]):  # Skip input layer
            x = layer(x)
        
        # Buat model baru
        new_model = tf.keras.Model(inputs=inputs, outputs=x)
        
        return new_model
        
    except Exception as e:
        st.error(f"Gagal membuat wrapper model: {str(e)}")
        return None

def load_model():
    """Load model dengan berbagai strategi fallback"""
    st.info("🔄 Mencoba memuat model...")
    
    model_path = 'models/model_potato.keras'
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file tidak ditemukan: {model_path}")
        return create_dummy_model()
    
    # Strategi 1: Load model langsung
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model berhasil dimuat langsung!")
        return model
    except Exception as e1:
        st.warning(f"⚠️ Gagal load langsung: {str(e1)}")
        
        # Strategi 2: Load dengan custom objects
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            st.success("✅ Model berhasil dimuat tanpa compile!")
            return model
        except Exception as e2:
            st.warning(f"⚠️ Gagal load tanpa compile: {str(e2)}")
            
            # Strategi 3: Buat wrapper model
            try:
                model = rebuild_model_wrapper()
                if model is not None:
                    st.success("✅ Model wrapper berhasil dibuat!")
                    return model
            except Exception as e3:
                st.warning(f"⚠️ Gagal buat wrapper: {str(e3)}")
    
    # Strategi 4: Model dummy sebagai fallback
    st.error("❌ Semua strategi loading gagal. Menggunakan model dummy.")
    return create_dummy_model()

def create_dummy_model():
    """Buat model dummy yang berfungsi untuk testing"""
    st.info("🔧 Membuat model dummy...")
    
    try:
        # Model dummy sederhana tapi realistis
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(300, 300, 3)),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ Model dummy berhasil dibuat!")
        return model
        
    except Exception as e:
        st.error(f"❌ Gagal membuat model dummy: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess gambar - selalu ke RGB 300x300"""
    try:
        # Selalu convert ke RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ke 300x300
        img = image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Convert ke numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Pastikan shape (300, 300, 3)
        if img_array.shape != (300, 300, 3):
            st.error(f"❌ Unexpected shape after resize: {img_array.shape}")
            return None
        
        # Normalisasi [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {str(e)}")
        return None

def predict_disease(model, image, is_dummy=False):
    """Prediksi penyakit dari gambar"""
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None, None
        
        # Prediksi
        predictions = model.predict(processed_image, verbose=0)
        
        # Class names
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        
        if is_dummy:
            # Untuk model dummy, buat prediksi random yang masuk akal
            np.random.seed(42)  # Untuk konsistensi
            fake_predictions = np.random.dirichlet([1, 1, 2])  # Bias ke healthy
            predictions = np.array([fake_predictions])
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        return class_names[predicted_class], confidence, predictions[0]
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None, None, None

def save_detection_history(username, prediction, confidence, image_path):
    """Simpan riwayat deteksi"""
    try:
        history_file = 'data/history.json'
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        new_detection = {
            'username': username,
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'image_path': image_path
        }
        
        history.append(new_detection)
        
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=4)
            
    except Exception as e:
        # Silent fail untuk deployment
        pass

def show_detection():
    st.title("🔍 Deteksi Penyakit Daun Kentang")
    st.markdown("---")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Tidak dapat memuat model. Aplikasi tidak dapat berjalan.")
        st.stop()
    
    # Deteksi apakah menggunakan dummy model
    is_dummy = hasattr(model, 'layers') and len(model.layers) < 10
    
    if is_dummy:
        st.warning("⚠️ **Menggunakan Model Demo** - Hasil prediksi bersifat simulasi untuk testing aplikasi.")
    
    class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    
    st.write("Upload gambar daun kentang untuk mendeteksi penyakit:")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Pilih gambar...",
        type=['png', 'jpg', 'jpeg'],
        help="Upload gambar daun kentang dalam format PNG, JPG, atau JPEG"
    )
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Gambar Input")
                st.image(image, caption="Gambar yang diupload", use_column_width=True)
            
            with col2:
                st.subheader("Hasil Deteksi")
                
                if st.button("🔍 Analisis Gambar", use_container_width=True):
                    with st.spinner("Menganalisis gambar..."):
                        prediction, confidence, all_predictions = predict_disease(model, image, is_dummy)
                        
                        if prediction is not None:
                            formatted_prediction = prediction.replace("Potato___", "").replace("_", " ").title()
                            
                            # Display results
                            clean_pred = prediction.replace("Potato___", "").replace("_", " ").title()
                            
                            if "healthy" in prediction.lower():
                                st.success(f"🌿 **Hasil: {clean_pred}**")
                                if not is_dummy:
                                    st.balloons()
                            elif "early" in prediction.lower():
                                st.warning(f"⚠️ **Hasil: {clean_pred}**")
                            else:
                                st.error(f"🚨 **Hasil: {clean_pred}**")
                            
                            st.write(f"**Confidence Score:** {confidence:.2f}%")
                            st.progress(confidence / 100)
                            
                            # Show all probabilities
                            with st.expander("Detail Probabilitas"):
                                for class_name, prob in zip(class_names, all_predictions):
                                    label = class_name.replace("Potato___", "").replace("_", " ").title()
                                    st.write(f"**{label}:** {prob*100:.2f}%")
                            
                            if is_dummy:
                                st.info("ℹ️ **Catatan:** Ini adalah hasil simulasi dari model demo.")
                            
                            # Save history
                            try:
                                username = st.session_state.get('username', 'anonymous')
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                image_path = f"temp_{username}_{timestamp}.jpg"
                                save_detection_history(username, formatted_prediction, confidence, image_path)
                            except:
                                pass  # Silent fail
                        else:
                            st.error("❌ Gagal menganalisis gambar.")
                            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    st.markdown("---")
    st.info("💡 **Tips:** Untuk hasil terbaik, gunakan gambar dengan pencahayaan yang baik dan fokus pada daun yang ingin dianalisis.")
    
    # Troubleshooting info
    with st.expander("🔧 Solusi Masalah Model"):
        st.markdown("""
        **Error yang terjadi:** Model Anda disimpan dengan input shape yang salah.
        
        **Penyebab:** 
        - Model expect input `(None, 301, 301, 1)` - grayscale 301x301
        - Tapi layer EfficientNet expect `(None, ?, ?, 3)` - RGB
        
        **Solusi Definitif - Re-train model Anda:**
        
        ```python
        # 1. Pastikan input layer yang benar
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=(300, 300, 3),  # RGB, bukan grayscale!
            include_top=False,
            weights='imagenet'
        )
        
        # 2. Bangun model dengan benar
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        # 3. Compile dan train ulang
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # 4. Simpan dengan benar
        model.save('models/model_potato.keras')
        ```
        
        **Sementara ini, aplikasi menggunakan model demo untuk testing.**
        """)
