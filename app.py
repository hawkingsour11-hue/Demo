import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

st.set_page_config(page_title="Drought Prediction", layout="wide")

st.title("Drought Prediction Dashboard")
st.text("Input environmental metrics, coordinates, and seasonal data to assess localized drought probability.")

# 1. Rebuild the precise architecture dumped by your error log
@st.cache_resource
def load_artifacts():
    # Structural blueprint reconstruction matching your Keras Tuner config exactly
    model = tf.keras.models.Sequential([
        # Input + Layer 0
        tf.keras.layers.Dense(64, activation='relu', input_shape=(15,), name='dense'),
        # Layer 1
        tf.keras.layers.Dropout(0.4, name='dropout'),
        tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
        # Layer 2
        tf.keras.layers.Dense(96, activation='relu', name='dense_2'),
        # Layer 3
        tf.keras.layers.Dense(160, activation='relu', name='dense_3'),
        # Layer 4
        tf.keras.layers.Dense(128, activation='relu', name='dense_4'),
        # Output Layer
        tf.keras.layers.Dense(1, activation='sigmoid', name='dense_5')
    ])
    
    # Inject the raw weight matrices safely
    model.load_weights('best_drought_weights.weights.h5')
    
    # Load your standard scaling parameters
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading deployment files: {e}")
    st.stop()

# 2. Organize your 15 exact feature names into logical UI sections
st.subheader("Model Input Variables")
col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Weather & Climate**")
    t2m = st.number_input("Average Temperature (T2M)", value=25.0, format="%.2f")
    t2m_max = st.number_input("Max Temperature (T2M_MAX)", value=32.0, format="%.2f")
    t2m_min = st.number_input("Min Temperature (T2M_MIN)", value=18.0, format="%.2f")
    rh2m = st.number_input("Relative Humidity (RH2M)", value=60.0, format="%.2f")
    ws2m = st.number_input("Wind Speed at 2 Meters (WS2M)", value=3.5, format="%.2f")

with col2:
    st.write("**Environment & Location**")
    allsky = st.number_input("Solar Radiation (ALLSKY_SFC_SW_DWN)", value=15.0, format="%.2f")
    prectotcorr = st.number_input("Precipitation Corrected (PRECTOTCORR)", value=2.0, format="%.2f")
    spei = st.number_input("Standardized Precipitation Index (spei)", value=0.0, format="%.2f")
    row_id = st.number_input("Row Identifier (row_id)", value=1, step=1)

with col3:
    st.write("**Cyclical Geospatial / Time Data**")
    lat_sin = st.number_input("Latitude Sine Transformation", value=0.0, format="%.4f")
    lat_cos = st.number_input("Latitude Cosine Transformation", value=1.0, format="%.4f")
    lon_sin = st.number_input("Longitude Sine Transformation", value=0.0, format="%.4f")
    lon_cos = st.number_input("Longitude Cosine Transformation", value=1.0, format="%.4f")
    month_sin = st.number_input("Month Sine Transformation", value=0.5, format="%.4f")
    month_cos = st.number_input("Month Cosine Transformation", value=0.8, format="%.4f")

# 3. Align array format to the exact original training columns order
ordered_features = [
    row_id, rh2m, t2m_max, t2m_min, ws2m, t2m, 
    allsky, prectotcorr, spei, 
    lat_sin, lat_cos, lon_sin, lon_cos, 
    month_sin, month_cos
]

# 4. Process Inputs and Execute Model Inference
if st.button("Analyze Drought Risk", type="primary", use_container_width=True):
    raw_array = np.array([ordered_features], dtype=np.float32)
    scaled_array = scaler.transform(raw_array)
    
    with st.spinner("Processing deep learning model weights..."):
        prediction_arr = model.predict(scaled_array)
        prediction_prob = float(prediction_arr)
    
    st.subheader("Risk Assessment Result")
    if prediction_prob >= 0.70:
        st.error(f"High Drought Risk Level Detected: {prediction_prob:.2%}")
    elif prediction_prob >= 0.40:
        st.warning(f"Moderate Drought Risk Level Detected: {prediction_prob:.2%}")
    else:
        st.success(f"Low / Negligible Drought Risk Level: {prediction_prob:.2%}")
