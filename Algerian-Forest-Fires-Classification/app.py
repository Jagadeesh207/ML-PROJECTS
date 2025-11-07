import streamlit as st
import pandas as pd
import joblib
import base64
import os  # <-- Using os for robust path handling

# ----------------------------------------
# 1️⃣ PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="🔥 Forest Fire Risk Prediction",
    page_icon="🔥",
    layout="centered"
)

# ----------------------------------------
# 2️⃣ CUSTOM CSS WITH LOCAL IMAGE
# ----------------------------------------

# FUNCTION TO LOAD AND ENCODE IMAGE
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


# --- BUILD A ROBUST FILE PATH ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Using the filename you provided ---
YOUR_IMAGE_FILENAME = "forest-fire-for-kishan-img-jpg.webp" 

IMG_FILE_PATH = os.path.join(SCRIPT_DIR, YOUR_IMAGE_FILENAME)


# Check if file exists using os.path.isfile
if not os.path.isfile(IMG_FILE_PATH):
    st.error(f"Image file not found at: {IMG_FILE_PATH}")
    st.warning("Please check the `YOUR_IMAGE_FILENAME` variable in your code and make sure it's spelled correctly.")
    img_base64 = ""
else:
    img_base64 = get_img_as_base64(IMG_FILE_PATH)


# 3. CUSTOM CSS (Corrected: All braces are doubled '{{' and '}}')
st.markdown(f"""
<style>
.stApp {{
    background-image: url('data:image/webp;base64,{img_base64}');
    background-size: cover;
    background-attachment: fixed;
    background-position: center center;
    color: #ffffff;
    font-family: 'Poppins', sans-serif;
}}

.stApp::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    z-index: -1;
}}

h1 {{
    text-align: center;
    font-size: 3rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.6);
    margin-bottom: 0.5rem;
}}

h2, h3, h5, .stSubheader, h5 {{
    color: #ffffff !important;
    text-shadow: 2px 2px 8px rgba(0,0,0,0.7);
}}

.subtitle {{
    text-align: center;
    font-size: 1.1rem;
    color: rgba(255,255,255,0.95);
    margin-bottom: 2rem;
    text-shadow: 1px 1px 6px rgba(0,0,0,0.7);
}}

input[type="number"] {{
    color: #FFFFFF !important;
}}

div[data-baseweb="select"] > div, 
.stNumberInput > div > div > input {{
    background-color: rgba(0, 0, 0, 0.4);
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
}}

.stSlider > label, 
.stSelectbox > label, 
.stNumberInput > label {{
    color: #ffffff !important;
    font-weight: 600;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
}}

div[data-baseweb="slider"] > div:first-child {{
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.2);
}}

div[data-baseweb="slider"] > div:nth-child(2) {{
    background-color: #ff512f;
}}

.stSlider .stNumberInput > div > div > input {{
    background-color: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.2);
}}

.stButton > button {{
    background: linear-gradient(135deg, #ff512f, #dd2476);
    color: white;
    font-size: 1.2rem;
    font-weight: 700;
    border: none;
    border-radius: 50px;
    padding: 0.75rem 2rem;
    width: 100%;
    box-shadow: 0 8px 15px rgba(221,36,118,0.4);
}}

.stButton > button:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 25px rgba(221,36,118,0.6);
}}

.result-card {{
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin-top: 2rem;
}}

.result-fire {{
    background: linear-gradient(135deg, #f85032, #e73827);
    color: white;
}}

.result-safe {{
    background: linear-gradient(135deg, #1D976C, #93F9B9);
    color: #000;
}}

/* --- NEW: Style for the st.info box --- */
.stAlert {{
    background-color: rgba(0, 0, 0, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
}}

.stAlert p {{
    color: #ffffff;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}}
/* --- End of new styles --- */

footer, #MainMenu, header {{
    visibility: hidden;
}}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# 3️⃣ LOAD MODELS
# ----------------------------------------
@st.cache_resource
def load_models():
    try:
        # This now correctly uses SCRIPT_DIR
        models = {
            "scaler": joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl")),
            "onehot": joblib.load(os.path.join(SCRIPT_DIR, "onehot.pkl")),
            "pca": joblib.load(os.path.join(SCRIPT_DIR, "pca.pkl")),
            "model": joblib.load(os.path.join(SCRIPT_DIR, "model.pkl")),
            "label_encoder": joblib.load(os.path.join(SCRIPT_DIR, "labelencoder.pkl"))
        }
        return models
    except Exception as e:
        st.error(f"Error loading model files from {SCRIPT_DIR}: {e}")
        st.stop()

models = load_models()
NUMERIC_COLS = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI']
CAT_COLS = ['Region']

# ----------------------------------------
# 4️⃣ PREDICTION FUNCTION
# ----------------------------------------
def make_prediction(input_data):
    try:
        scaler = models['scaler']
        onehot = models['onehot']
        pca = models['pca']
        model = models['model']
        label_encoder = models['label_encoder']

        input_df = pd.DataFrame([input_data])
        num_df = input_df[NUMERIC_COLS]

        cat_df_transformed = onehot.transform(input_df[CAT_COLS])
        cat_cols = onehot.get_feature_names_out(CAT_COLS)
        cat_df = pd.DataFrame(cat_df_transformed, columns=cat_cols)

        X = pd.concat([num_df.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)
        X = X[scaler.feature_names_in_] 
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        y_pred = model.predict(X_pca)
        return label_encoder.inverse_transform(y_pred.astype(int))[0]

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ----------------------------------------
# 5️⃣ TITLE
# ----------------------------------------
st.markdown("<h1>🔥 Forest Fire Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-driven analysis of climatic variables for proactive fire management.</p>', unsafe_allow_html=True)

# ----------------------------------------
# 6️⃣ INPUTS
# ----------------------------------------
st.subheader("🌦️ Input Parameters")

# Weather inputs
col1, col2 = st.columns(2)
with col1:
    region = st.selectbox("Region", ["A", "B"])
with col2:
    temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0, step=0.1)

col3, col4, col5 = st.columns(3)
with col3:
    rh = st.number_input("Relative Humidity (%)", 0.0, 100.0, 50.0, step=0.1)
with col4:
    ws = st.number_input("Wind Speed (km/h)", 0.0, 100.0, 10.0, step=0.1)
with col5:
    rain = st.number_input("Rain (mm)", 0.0, 50.0, 0.0, step=0.01)

st.markdown("---")
st.markdown("##### 🔥 FWI System Indices")

# Initialize session state
for col in ["ffmc", "dmc", "dc", "isi", "bui", "fwi"]:
    if col not in st.session_state:
        st.session_state[col] = 50.0

# FWI sliders
cols = st.columns(3)
for i, key in enumerate(["ffmc", "dmc", "dc"]):
    with cols[i]:
        st.session_state[key] = st.slider(key.upper(), 0.0, 100.0, st.session_state[key])
        st.number_input(f"{key.upper()} Value", 0.0, 100.0, value=st.session_state[key], step=0.1)

cols2 = st.columns(3)
for i, key in enumerate(["isi", "bui", "fwi"]):
    with cols2[i]:
        st.session_state[key] = st.slider(key.upper(), 0.0, 100.0, st.session_state[key])
        st.number_input(f"{key.upper()} Value", 0.0, 100.0, value=st.session_state[key], step=0.1)

st.markdown("---")

# ----------------------------------------
# 7️⃣ PREDICTION
# ----------------------------------------
if st.button("🚀 Predict Fire Risk"):
    input_data = {
        "Region": region,
        "Temperature": temp,
        "RH": rh,
        "Ws": ws,
        "Rain": rain,
        "FFMC": st.session_state.ffmc,
        "DMC": st.session_state.dmc,
        "DC": st.session_state.dc,
        "ISI": st.session_state.isi,
        "BUI": st.session_state.bui,
        "FWI": st.session_state.fwi
    }

    with st.spinner("Analyzing data... 🔍"):
        result = make_prediction(input_data)

    if result and "fire" in result.lower():
        st.markdown("""
        <div class="result-card result-fire">
            <h2>🔥 HIGH RISK DETECTED</h2>
            <p>The model predicts a potential <strong>Forest Fire</strong> under current conditions.</p>
            <p>🚨 Take precautionary measures immediately!</p>
        </div>
        """, unsafe_allow_html=True)
    elif result:
        st.markdown("""
        <div class="result-card result-safe">
            <h2>🌿 LOW RISK</h2>
            <p>Conditions are safe — <strong>No fire risk</strong> predicted at this time.</p>
            <p>✅ Continue monitoring weather conditions regularly.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Adjust the input parameters above and click **Predict Fire Risk** to see results.")