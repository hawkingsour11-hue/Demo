import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

st.set_page_config(page_title="Drought Prediction", layout="wide")

# Custom CSS styling for a clean dashboard look
st.markdown("""
    <style>
    .main {background-color: #f7f9fb;}
    h1 {color: #2e7d32;}
    h3 {color: #1b5e20; margin-top: 20px;}
    </style>
""", unsafe_allowed_html=True)

st.title("🌾 NASA POWER Drought Prediction Dashboard")
st.write("Input environmental metrics, coordinates, and seasonal data to assess localized drought probability.")

# 1. Load the Model and Scaler using Keras 3 version compatibility
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model('best_drought_model.keras')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Error loading deployment files: {e}")
    st.stop()

# 2. Organize your 15 exact feature names into logical UI sections
st.subheader("📊 Model Input Variables")

# Create 3 clean vertical visual columns for the user interface
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🌡️ Weather & Climate")
    t2m = st.number_input("Average Temperature (T2M)", value=25.0, format="%.2f")
    t2m_max = st.number_input("Max Temperature (T2M_MAX)", value=32.0, format="%.2f")
    t2m_min = st.number_input("Min Temperature (T2M_MIN)", value=18.0, format="%.2f")
    rh2m = st.number_input("Relative Humidity (RH2M)", value=60.0, format="%.2f")
    ws2m = st.number_input("Wind Speed at 2 Meters (WS2M)", value=3.5, format="%.2f")

with col2:
    st.markdown("### ☀️ Environment & Location")
    allsky = st.number_input("Solar Radiation (ALLSKY_SFC_SW_DWN)", value=15.0, format="%.2f")
    prectotcorr = st.number_input("Precipitation Corrected (PRECTOTCORR)", value=2.0, format="%.2f")
    spei = st.number_input("Standardized Precipitation Index (spei)", value=0.0, format="%.2f")
    row_id = st.number_input("Row Identifier (row_id)", value=1, step=1)
    st.write("") # Spacer

with col3:
    st.markdown("### 🌐 Cyclical Geospatial / Time Data")
    lat_sin = st.number_input("Latitude Sine Transformation", value=0.0, format="%.4f")
    lat_cos = st.number_input("Latitude Cosine Transformation", value=1.0, format="%.4f")
    lon_sin = st.number_input("Longitude Sine Transformation", value=0.0, format="%.4f")
    lon_cos = st.number_input("Longitude Cosine Transformation", value=1.0, format="%.4f")
    month_sin = st.number_input("Month Sine Transformation", value=0.5, format="%.4f")
    month_cos = st.number_input("Month Cosine Transformation", value=0.8, format="%.4f")

# 3. Construct the array in the EXACT order of your original X dataframe columns
# Order: 'row_id', 'RH2M', 'T2M_MAX', 'T2M_MIN', 'WS2M', 'T2M', 'ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'spei', 'lat_sin', 'lat_cos', 'lon_sin', 'lon_cos', 'month_sin', 'month_cos'
ordered_features = [
    row_id, rh2m, t2m_max, t2m_min, ws2m, t2m, 
    allsky, prectotcorr, spei, 
    lat_sin, lat_cos, lon_sin, lon_cos, 
    month_sin, month_cos
]

st.markdown("---")

# 4. Run Preprocessing and Prediction pipeline
if st.button("🔮 Analyze Drought Risk", type="primary", use_container_width=True):
    # Format input as a 2D matrix: shape (1, 15)
    raw_array = np.array([ordered_features], dtype=np.float32)
    
    # Scale inputs using your original training metrics
    scaled_array = scaler.transform(raw_array)
    
    with st.spinner("Processing deep learning model weights..."):
        # run prediction model matrix multiplication
        prediction_prob = model.predict(scaled_array)[0][0]
    
    st.subheader("📋 Risk Assessment Result")
    
    # Render interactive outputs depending on continuous risk levels
    if prediction_prob >= 0.70:
        st.error(f"🚨 **High Drought Risk Level Detected:** {prediction_prob:.2%}")
        st.warning("Action advised: Implement localized agricultural water conservation protocols.")
    elif prediction_prob >= 0.40:
        st.warning(f"⚠️ **Moderate Drought Risk Level Detected:** {prediction_prob:.2%}")
        st.info("Action advised: Monitor soil degradation profiles closely over upcoming cyclical phases.")
    else:
        st.success(f"✅ **Low / Negligible Drought Risk Level:** {prediction_prob:.2%}")
        st.write("Current environmental indices indicate standard ecosystem stability.")
