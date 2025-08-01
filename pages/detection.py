import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime
import json

def load_model():
    """Load model keras dengan validasi"""
    try:
        # Debugging: print TensorFlow version
        st.info(f"TensorFlow version: {tf.__version__}")
        
        model_path = 'models/model_potato.keras'
        if not os.path.exists(model_path):
            st.error(f"Model file tidak ditemukan: {model_path}")
            return None
            
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Debug: Print model info
        st.info("✅ Model berhasil dimuat!")
        
        # Tampilkan informasi input shape
        input_shape = model.input_shape
        st.info(f"Model input shape: {input_shape}")
        
        # Validasi input shape
        if len(input_shape) != 4:
            st.error(f"❌ Model input shape tidak valid: {input_shape}")
            return None
            
        expected_channels = input_shape[-1] if input_shape[-1] is not None else 3
        expected_height = input_shape[1] if input_shape[1] is not None else 300
        expected_width = input_shape[2] if input_shape[2] is not None else 300
        
        st.info(f"Expected dimensions: {expected_height}x{expected_width}x{expected_channels}")
        
        # Store expected dimensions in session state
        st.session_state.model_height = expected_height
        st.session_state.model_width = expected_width  
        st.session_state.model_channels = expected_channels
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.error("Kemungkinan penyebab:")
        st.error("1. File model corrupt atau tidak kompatibel")
        st.error("2. Model disimpan dengan TensorFlow versi berbeda")
        st.error("3. Model tidak disimpan dengan benar")
        
        # Suggestion untuk rebuild model
        st.info("💡 **Solusi yang disarankan:**")
        st.info("1. Re-train dan simpan ulang model dengan TensorFlow versi yang sama")
        st.info("2. Pastikan input shape saat training adalah (None, 300, 300, 3)")
        st.info("3. Gunakan model.save() alih-alih model.save_weights()")
        
        return None

def preprocess_image(image):
    """Preprocess gambar untuk prediksi - DINAMIS berdasarkan model"""
    try:
        # Get expected dimensions from session state
        height = st.session_state.get('model_height', 300)
        width = st.session_state.get('model_width', 300)
        channels = st.session_state.get('model_channels', 3)
        
        st.info(f"Preprocessing untuk: {height}x{width}x{channels}")
        
        # Convert berdasarkan expected channels
        if channels == 1:
            # Model expects grayscale
            if image.mode != 'L':
                image = image.convert('L')
        else:
            # Model expects RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
        
        # Resize ke ukuran yang diharapkan model
        img = image.resize((width, height), Image.Resampling.LANCZOS)
        
        # Convert ke numpy array
        img_array = np.array(img, dtype=np.float32)
        
        # Handle different channel requirements
        if channels == 1:
            # Grayscale
            if len(img_array.shape) == 3:
                img_array = np.mean(img_array, axis=-1)  # Convert to grayscale
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        else:
            # RGB
            if len(img_array.shape) == 2:  # Grayscale input
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[-1] == 1:  # Single channel
                img_array = np.repeat(img_array, 3, axis=-1)
            elif img_array.shape[-1] == 4:  # RGBA
                img_array = img_array[:, :, :3]  # Take RGB only
        
        # Final shape validation
        expected_shape = (height, width, channels)
        if img_array.shape != expected_shape:
            st.error(f"❌ Shape mismatch: got {img_array.shape}, expected {expected_shape}")
            return None
            
        st.success(f"✅ Preprocessing berhasil: {img_array.shape}")
        
        # Normalisasi
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        st.error(f"❌ Error in preprocessing: {str(e)}")
        return None

def create_dummy_model():
    """Buat model dummy untuk testing jika model asli bermasalah"""
    st.warning("🔧 Membuat model dummy untuk testing...")
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(300, 300, 3)),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    st.info("✅ Model dummy berhasil dibuat dengan input shape (None, 300, 300, 3)")
    return model

def predict_disease(model, image):
    """Prediksi penyakit dari gambar"""
    try:
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, None, None
            
        st.info(f"Input ke model: {processed_image.shape}")
        
        predictions = model.predict(processed_image, verbose=0)
        
        # Class names
        class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
        
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]) * 100)
        
        return class_names[predicted_class], confidence, predictions[0]
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.error("Detail error untuk debugging:")
        st.code(str(e))
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
        st.warning(f"Could not save history: {str(e)}")

def show_detection():
    st.title("🔍 Deteksi Penyakit Daun Kentang")
    st.markdown("---")
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=True)
    use_dummy = st.sidebar.checkbox("🔧 Use Dummy Model (for testing)", value=False)
    
    # Load model
    if use_dummy:
        model = create_dummy_model()
    else:
        model = load_model()
        
    if model is None and not use_dummy:
        st.error("❌ Model tidak dapat dimuat. Coba aktifkan 'Use Dummy Model' untuk testing.")
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
            
            if debug_mode:
                st.info(f"📊 Info gambar: {image.size}, mode: {image.mode}")
            
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
                            formatted_prediction = prediction.replace("Potato___", "").replace("_", " ").title()
                            
                            # Save image
                            try:
                                os.makedirs("uploads", exist_ok=True)
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                username = st.session_state.get('username', 'user')
                                image_filename = f"{username}_{timestamp}.jpg"
                                image_path = f"uploads/{image_filename}"
                                image.save(image_path)
                            except Exception as e:
                                if debug_mode:
                                    st.warning(f"Could not save image: {str(e)}")
                                image_path = "temp_image.jpg"
                            
                            # Display results
                            clean_pred = prediction.replace("Potato___", "").replace("_", " ").title()
                            
                            if "healthy" in prediction.lower():
                                st.success(f"🌿 **Hasil: {clean_pred}**")
                                if not use_dummy:  # Only show balloons for real predictions
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
                            
                            # Save history
                            username = st.session_state.get('username', 'anonymous')
                            save_detection_history(username, formatted_prediction, confidence, image_path)
                            
                            if use_dummy:
                                st.info("ℹ️ Ini adalah hasil dari model dummy untuk testing.")
                        else:
                            st.error("❌ Gagal menganalisis gambar.")
                            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            if debug_mode:
                st.code(str(e))
    
    st.markdown("---")
    st.info("💡 **Tips:** Untuk hasil terbaik, gunakan gambar dengan pencahayaan yang baik dan fokus pada daun yang ingin dianalisis.")
    
    # Model troubleshooting section
    with st.expander("🔧 Model Troubleshooting"):
        st.markdown("""
        **Jika model tidak dapat dimuat:**
        
        1. **Re-train model dengan kode ini:**
        ```python
        # Pastikan input shape yang benar
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(300, 300, 3)),  # RGB input
            # ... layer lainnya
        ])
        
        # Simpan dengan format yang benar
        model.save('models/model_potato.keras')
        ```
        
        2. **Periksa versi TensorFlow:**
           - Training: TensorFlow versi berapa?
           - Deployment: Lihat info di atas
        
        3. **Test dengan dummy model** untuk memastikan kode berfungsi
        """)
