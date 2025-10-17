import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Use caching to load the models and encoders only once
@st.cache_resource
def load_artifacts():
    """
    Loads the saved model, scaler, and one-hot encoder from pickle files.
    """
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('onehot.pkl', 'rb') as onehot_file:
        onehot_encoder = pickle.load(onehot_file)
    return model, scaler, onehot_encoder

# Load the artifacts
model, scaler, onehot_encoder = load_artifacts()

# Set up the Streamlit page
st.set_page_config(page_title="Abalone Age Prediction", layout="wide")
st.title("🐚 Abalone Age Prediction")
st.write("Enter the physical measurements of an abalone to predict its age.")

# Create the user input interface in the sidebar
st.sidebar.header("Input Abalone Features")

def user_input_features():
    """
    Creates sliders and select boxes for user input.
    """
    sex = st.sidebar.selectbox('Sex', ('M', 'F', 'I'))
    length = st.sidebar.slider('Length (mm)', 0.0, 1.0, 0.5)
    diameter = st.sidebar.slider('Diameter (mm)', 0.0, 1.0, 0.4)
    height = st.sidebar.slider('Height (mm)', 0.0, 0.5, 0.15)
    whole_weight = st.sidebar.slider('Whole Weight (grams)', 0.0, 3.0, 1.0)
    shucked_weight = st.sidebar.slider('Shucked Weight (grams)', 0.0, 1.5, 0.5)
    viscera_weight = st.sidebar.slider('Viscera Weight (grams)', 0.0, 0.8, 0.2)
    shell_weight = st.sidebar.slider('Shell Weight (grams)', 0.0, 1.0, 0.3)

    data = {
        'Sex': sex,
        'Length': length,
        'Diameter': diameter,
        'Height': height,
        'Whole weight': whole_weight,
        'Whole weight.1': shucked_weight, # Corresponds to Shucked weight
        'Whole weight.2': viscera_weight, # Corresponds to Viscera weight
        'Shell weight': shell_weight
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display the user's input
st.subheader("Your Input Measurements")
st.write(input_df)

# Create a button to make predictions
if st.button("Predict Age"):
    # --- PREPROCESSING PIPELINE ---
    # This must be IDENTICAL to your training pipeline

    # 1. Separate categorical and numerical features from the input
    input_cat = input_df[['Sex']]
    input_num = input_df.drop('Sex', axis=1)

    # 2. Apply the one-hot encoder to the 'Sex' column
    encoded_sex = onehot_encoder.transform(input_cat)
    encoded_sex_df = pd.DataFrame(encoded_sex, columns=onehot_encoder.get_feature_names_out(['Sex']))

    # 3. Concatenate the one-hot encoded columns and the numerical columns
    # Ensure the numerical columns are in the same order as during training
    concat_df = pd.concat([input_num.reset_index(drop=True), encoded_sex_df.reset_index(drop=True)], axis=1)
    
    # 4. Apply the scaler to the entire concatenated DataFrame
    scaled_df = scaler.transform(concat_df)

    # --- PREDICTION ---
    # 5. Make a prediction using the final model
    predicted_rings = model.predict(scaled_df)
    predicted_age = predicted_rings[0] + 1.5 # Age = Rings + 1.5

    # Display the result
    st.success(f"**Predicted Age of Abalone: {predicted_age:.1f} years**")