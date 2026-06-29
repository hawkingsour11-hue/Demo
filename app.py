import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

st.set_page_config(page_title="Drought Prediction", layout="centered")
st.title("🌾 Drought Prediction Dashboard")
st.write("Input environmental and soil metrics to predict drought probability.")

# 1. Load the Model and the Scaler safely
@st.cache_resource
def load_artifacts():
    # 1. Rebuild the precise architectural blueprint manually
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(15,)), # Match your exact feature count
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 2. Seamlessly inject the saved mathematical weights into the blueprint
    model.load_weights('best_drought_weights.weights.h5')
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading deployment files: {e}")
    st.stop()

# 2. Define your Exact Dataset Feature Names
# REPLACE these string placeholders with your actual X.columns output from Colab
FEATURE_NAMES = ["Temperature", "Humidity", "Precipitation", "Soil Moisture", "Wind Speed"]

st.subheader("📊 Environmental Metrics Input")
user_inputs = []

# Dynamically generate clean layout columns for your inputs
cols = st.columns(2)
for idx, name in enumerate(FEATURE_NAMES):
    # Alternates fields across 2 visual columns
    with cols[idx % 2]:
        val = st.number_input(f"{name}", value=0.0, step=0.1, format="%.2f")
        user_inputs.append(val)

# 3. Run Preprocessing and Prediction pipeline
if st.button("🔮 Analyze Drought Risk", type="primary"):
    # Convert inputs to 2D shape format: (1, num_features)
    raw_array = np.array([user_inputs])
    
    # CRITICAL STEP: Scale raw inputs identically to training pipeline
    scaled_array = scaler.transform(raw_array)
    
    with st.spinner("Processing network weights..."):
        prediction = model.predict(scaled_array)[0][0]
    
    st.subheader("Risk Assessment Result")
    
    # Visual gauging of risk levels
    if prediction >= 0.7:
        st.error(f"⚠️ High Drought Risk Level: {prediction:.2%}")
    elif prediction >= 0.4:
        st.warning(f"⚡ Moderate Drought Risk Level: {prediction:.2%}")
    else:
        st.success(f"✅ Low/Safe Drought Risk Level: {prediction:.2%}")
