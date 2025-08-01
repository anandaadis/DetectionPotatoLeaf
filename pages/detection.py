import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json

def fix_model_input():
    """Perbaiki model dengan membuat wrapper yang mengatasi masalah input shape"""
    try:
        st.info("🔧 Mencoba memperbaiki model dengan wrapper...")
        
        # Load model asli (meski ada error)
        model_path = 'models/model_potato.keras'
        
        # Strategi: Load weights saja, bukan full model
        try:
            # Coba load model secara paksa dengan custom handling
            original_model = tf.keras.models.load_model(model_path, compile=False)
            st.warning("⚠️ Model loaded tanpa compile, tapi mungkin masih ada issue...")
            return original_model
        except Exception as e:
            st.error(f"Model benar-benar tidak kompatibel: {str(e)}")
            
        # Jika gagal total, buat model baru dengan arsitektur yang sama
        st.info("🔄 Membuat model baru dengan arsitektur EfficientNetB3...")
        
        # Recreate EfficientNetB3 architecture dengan input yang benar
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=(300, 300, 3),
            include_top=False,
            weights='imagenet'
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        st.success("✅ Model baru berhasil dibuat dengan arsitektur yang benar!")
        st.warning("⚠️ Model ini menggunakan weights ImageNet (belum di-train untuk kentang)")
        
        return model
        
    except Exception as e:
        st.error(f"❌ Gagal membuat model: {str(e)}")
        return None

def load_model():
    """Load model dengan berbagai strategi"""
    model_path = 'models/model_potato.keras'
    
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan: {model_path}")
        return None
    
    st.info(f"📁 Model ditemukan: {model_path}")
    st.info(f"🔍 TensorFlow version: {tf.__version__}")
    
    # Strategi 1: Load langsung
    try:
        st.info("🔄 Mencoba load model langsung...")
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model berhasil dimuat langsung!")
        
        # Check input shape
        input_shape = model.input_shape
        st.info(f"📊 Model input shape: {input_shape}")
        return model
        
    except Exception as e:
        st.error(f"❌ Load langsung gagal: {str(e)}")
        
        # Strategi 2: Load tanpa compile
        try:
            st.info("🔄 Mencoba load tanpa compile...")
            model = tf.keras.models.load_model(model_path, compile=False)
            st.success("✅ Model dimuat tanpa compile!")
            
            input_shape = model.input_shape
            st.info(f"📊 Model input shape: {input_shape}")
            return model
            
        except Exception as e2:
            st.error(f"❌ Load tanpa compile gagal: {str(e2)}")
            
            # Strategi 3: Buat model baru
            return fix_model_input()

def preprocess_image_for_broken_model(image):
    """Preprocessing khusus untuk model yang bermasalah"""
    try:
        # Convert ke grayscale jika model expect 1 channel
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize ke 301x301 sesuai error message
        img = image.resize((301, 301), Image.Resampling.LANCZOS)
        
        # Convert ke numpy
        img_array = np.array(img, dtype=np.float32)
        
        # Add channel dimension untuk grayscale
        img_array = np.expand_dims(img_array, axis=-1)  # Shape: (301, 301, 1)
        
        # Normalisasi
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 301, 301, 1)
        
        st.info(f"🔧 Preprocessing untuk model bermasalah: {img_array.shape}")
        return img_array
        
    except Exception as e:
        st.error(f"❌ Error in special preprocessing: {str(e)}")
        return None

def preprocess_image_standard(image):
    """Preprocessing standard untuk model yang benar"""
    try:
        # Convert ke RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize ke 300x300
        img = image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Convert ke numpy
        img_array = np.array(img, dtype=np.float32)
        
        # Normalisasi
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        st.info(f"✅ Preprocessing standard: {img_array.shape}")
        return img_array
        
    except Exception as e:
        st.error(f"❌ Error in standard preprocessing: {str(e)}")
        return None

def predict_disease(model, image):
    """Prediksi dengan fallback preprocessing"""
    try:
        # Cek input shape model
        input_shape = model.input_shape
        st.info(f"🎯 Predicting dengan model input shape: {input_shape}")
        
        # Tentukan preprocessing berdasarkan input shape
        if len(input_shape) == 4:
            height, width, channels = input_shape[1], input_shape[2], input_shape[3]
            
            if channels == 1 and height == 301 and width == 301:
                # Model bermasalah - coba preprocessing khusus
                st.info("🔧 Menggunakan preprocessing untuk model bermasalah...")
                processed_image = preprocess_image_for_broken_model(image)
            else:
                # Model normal
                st.info("✅ Menggunakan preprocessing standard...")
                processed_image = preprocess_image_standard(image)
        else:
            # Default ke standard
            processed_image = preprocess_image_standard(image)
        
        if processed_image is None:
            return None, None, None
        
        # Prediksi
        st.info("🤖 Melakukan prediksi...")
        predictions = model.predict(processed_image, verbose=0)
        
        # Class names
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        st.success(f"✅ Prediksi berhasil: {class_names[predicted_class]} ({confidence:.2f}%)")
        
        return class_names[predicted_class], confidence, predictions[0]
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.code(str(e))  # Show full error for debugging
        return None, None, None

def save_detection_history(username, prediction, confidence, image_path):
    """Simpan riwayat deteksi"""
    try:
        history_file = 'data/history.json'
        os.makedirs('data', exist_ok=True)
        
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
        st.warning(f"Tidak dapat menyimpan history: {str(e)}")

def show_detection():
    st.title("🔍 Deteksi Penyakit Daun Kentang")
    st.markdown("---")
    
    # Debug info
    with st.expander("🔍 Debug Information", expanded=True):
        st.write(f"**TensorFlow Version:** {tf.__version__}")
        st.write(f"**Model Path:** models/model_potato.keras")
        st.write(f"**Model Exists:** {os.path.exists('models/model_potato.keras')}")
    
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Tidak dapat memuat model. Silakan periksa file model Anda.")
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
        try:
            image = Image.open(uploaded_file)
            
            st.info(f"📷 Gambar info: {image.size}, mode: {image.mode}")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Gambar Input")
                st.image(image, caption="Gambar yang diupload", use_column_width=True)
            
            with col2:
                st.subheader("Hasil Deteksi")
                
                if st.button("🔍 Analisis Gambar", use_container_width=True):
                    with st.spinner("Menganalisis gambar..."):
                        prediction, confidence, all_predictions = predict_disease(model, image)
                        
                        if prediction is not None:
                            # Format hasil
                            clean_pred = prediction.replace("Potato___", "").replace("_", " ").title()
                            
                            # Display results dengan emoji
                            if "healthy" in prediction.lower():
                                st.success(f"🌿 **Hasil: {clean_pred}**")
                                st.balloons()
                            elif "early" in prediction.lower():
                                st.warning(f"⚠️ **Hasil: {clean_pred}**")
                            else:
                                st.error(f"🚨 **Hasil: {clean_pred}**")
                            
                            st.write(f"**Confidence Score:** {confidence:.2f}%")
                            st.progress(confidence / 100)
                            
                            # Detail probabilitas
                            with st.expander("📊 Detail Probabilitas"):
                                for class_name, prob in zip(class_names, all_predictions):
                                    label = class_name.replace("Potato___", "").replace("_", " ").title()
                                    col_a, col_b = st.columns([3, 1])
                                    col_a.write(f"**{label}**")
                                    col_b.write(f"{prob*100:.2f}%")
                                    st.progress(prob)
                            
                            # Save history
                            try:
                                os.makedirs("uploads", exist_ok=True)
                                username = st.session_state.get('username', 'anonymous')
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                image_path = f"uploads/{username}_{timestamp}.jpg"
                                image.save(image_path)
                                
                                save_detection_history(username, clean_pred, confidence, image_path)
                                st.success("💾 Hasil disimpan ke history!")
                            except Exception as e:
                                st.warning(f"⚠️ Tidak dapat menyimpan: {str(e)}")
                        else:
                            st.error("❌ Gagal menganalisis gambar. Silakan coba lagi.")
                            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
    
    st.markdown("---")
    st.info("💡 **Tips:** Untuk hasil terbaik, gunakan gambar dengan pencahayaan yang baik dan fokus pada daun yang ingin dianalisis.")
    
    # Model info
    with st.expander("🔧 Informasi Model & Troubleshooting"):
        if model is not None:
            try:
                st.write(f"**Input Shape:** {model.input_shape}")
                st.write(f"**Output Shape:** {model.output_shape}")
                st.write(f"**Total Parameters:** {model.count_params():,}")
            except:
                st.write("Model info tidak tersedia")
        
        st.markdown("""
        **Jika masih ada error:**
        
        1. **Model Anda mungkin perlu di-retrain** dengan input shape yang benar
        2. **Gunakan kode ini untuk retrain:**
        
        ```python
        import tensorflow as tf
        
        # Model dengan input shape yang benar
        base_model = tf.keras.applications.EfficientNetB3(
            input_shape=(300, 300, 3),  # RGB 300x300
            include_top=False,
            weights='imagenet'
        )
        
        model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        # Train model dengan data Anda...
        model.save('models/model_potato.keras')
        ```
        """)
