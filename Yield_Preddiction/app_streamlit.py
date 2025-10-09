import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

MODELDIR = Path("./models")

# --- Function to load all models ---
@st.cache_resource
def load_artifacts():
    """Loads all the trained models and preprocessors."""
    try:
        preprocessor = joblib.load(MODELDIR / "preprocessor.joblib")
        le_crop = joblib.load(MODELDIR / "label_encoder.joblib")
        ohe = joblib.load(MODELDIR / "ohe_crop.joblib")
        rf_clf = joblib.load(MODELDIR / "rf_classifier.joblib")
        rf_reg = joblib.load(MODELDIR / "rf_regressor.joblib")
        # Load the .keras model, which is the modern format
        ann = tf.keras.models.load_model(MODELDIR / "ann_regressor_best.keras")
        return preprocessor, le_crop, ohe, rf_clf, rf_reg, ann
    except FileNotFoundError:
        return None, None, None, None, None, None

# --- Load Models ---
preprocessor, le_crop, ohe, rf_clf, rf_reg, ann = load_artifacts()

# --- Page Configuration ---
st.set_page_config(layout="centered", page_title="Crop & Yield Predictor")

# --- UI ---
st.title("🌾 Crop Recommendation & Yield Prediction System")

if not all([preprocessor, le_crop, ohe, rf_clf, rf_reg, ann]):
    st.error(f"Model files not found in the '{MODELDIR}' directory. Please run the `train.py` script first to generate them.")
else:
    st.markdown("Provide the soil and environmental conditions to get a crop recommendation and its predicted yield.")

    # Input widgets in the sidebar
    st.sidebar.header("Sensor Inputs")
    N = st.sidebar.slider("Nitrogen (N)", 0, 150, 90, help="Ratio of Nitrogen content in the soil")
    P = st.sidebar.slider("Phosphorus (P)", 5, 150, 45, help="Ratio of Phosphorus content in the soil")
    K = st.sidebar.slider("Potassium (K)", 5, 210, 45, help="Ratio of Potassium content in the soil")
    temperature = st.sidebar.slider("Temperature (°C)", 9.0, 44.0, 25.0, step=0.1, help="Temperature in degrees Celsius")
    humidity = st.sidebar.slider("Humidity (%)", 14.0, 100.0, 75.0, step=0.1, help="Relative humidity in percentage")
    rainfall = st.sidebar.slider("Rainfall (mm)", 20.0, 300.0, 100.0, step=0.1, help="Rainfall in mm")
    soil_moisture_input = st.sidebar.slider("Soil Moisture", 25.0, 85.0, 60.0, step=0.1, help="Moisture content of the soil")

    # Predict button
    if st.button("Recommend & Predict"):
        # Create a DataFrame with the correct column names as expected by the preprocessor
        X_input = pd.DataFrame([{
            "N": N,
            "P": P,
            "K": K,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "Soil_Moisture": soil_moisture_input  # Must match the training script
        }])

        # --- 1. Crop Recommendation ---
        # Scale the numeric features
        X_scaled = preprocessor.transform(X_input)
        
        # Predict the crop
        crop_pred_encoded = rf_clf.predict(X_scaled)[0]
        crop_pred_label = le_crop.inverse_transform([crop_pred_encoded])[0]

        st.subheader("Step 1: Recommended Crop")
        st.success(f"The most suitable crop for these conditions is: **{crop_pred_label.capitalize()}**")

        # --- 2. Yield Prediction ---
        # Prepare the input for the regression models
        # One-hot encode the predicted crop label
        crop_ohe = ohe.transform(np.array([crop_pred_label]).reshape(-1, 1))
        
        # Combine scaled features and the one-hot encoded crop
        X_reg_input = np.hstack([X_scaled, crop_ohe])

        # Predict yield with both models
        yield_rf = rf_reg.predict(X_reg_input)[0]
        yield_ann = ann.predict(X_reg_input, verbose=0).squeeze()

        st.subheader("Step 2: Predicted Yield")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Random Forest Prediction", value=f"{yield_rf:.2f} tons/ha")
        with col2:
            st.metric(label="Neural Network Prediction", value=f"{yield_ann:.2f} tons/ha")

        st.markdown("---")
        st.info("Disclaimer: Predictions are based on a synthetic dataset and are for demonstration purposes only. Real-world results may vary.")

    # Sidebar model info
    st.sidebar.markdown("---")
    st.sidebar.markdown("**✓ Models Loaded Successfully**")