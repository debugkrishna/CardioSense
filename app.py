import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cardiosense_model.pkl"

model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="CardioSense",
    page_icon="❤️",
    layout="centered"
)

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.section {
    font-size: 22px;
    font-weight: 600;
    margin-top: 20px;
    margin-bottom: 10px;
}

.disclaimer {
    font-size: 13px;
    color: #777;
    text-align: center;
    margin-top: 30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="title">❤️ CardioSense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Heart Disease Risk Prediction System</div>',
    unsafe_allow_html=True
)

st.write(
    "Enter the patient's clinical information below to generate "
    "a machine-learning-based risk prediction."
)

st.markdown(
    '<div class="section">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=50
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

st.markdown(
    '<div class="section">🫀 Cardiac Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    cp = st.selectbox(
        "Chest Pain Type",
        [
            "typical angina",
            "atypical angina",
            "non-anginal",
            "asymptomatic"
        ]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50.0,
        max_value=250.0,
        value=120.0
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50.0,
        max_value=700.0,
        value=200.0
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["False", "True", "Not provided"]
    )

with col2:
    restecg = st.selectbox(
        "Resting ECG",
        [
            "normal",
            "lv hypertrophy",
            "st-t abnormality",
            "Not provided"
        ]
    )

    thalch = st.number_input(
        "Maximum Heart Rate",
        min_value=50.0,
        max_value=250.0,
        value=150.0
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["False", "True", "Not provided"]
    )

    oldpeak = st.number_input(
        "ST Depression (Oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

st.markdown(
    '<div class="section">🔬 Additional Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    slope = st.selectbox(
        "ST Segment Slope",
        [
            "upsloping",
            "flat",
            "downsloping",
            "Not provided"
        ]
    )

with col2:
    ca = st.selectbox(
        "Number of Major Vessels",
        ["0", "1", "2", "3", "Not provided"]
    )

with col3:
    thal = st.selectbox(
        "Thalassemia",
        [
            "normal",
            "fixed defect",
            "reversable defect",
            "Not provided"
        ]
    )

fbs_value = np.nan if fbs == "Not provided" else fbs
restecg_value = np.nan if restecg == "Not provided" else restecg
exang_value = np.nan if exang == "Not provided" else exang
slope_value = np.nan if slope == "Not provided" else slope
thal_value = np.nan if thal == "Not provided" else thal
ca_value = np.nan if ca == "Not provided" else float(ca)

patient_data = pd.DataFrame({
    "age": [age],
    "sex": [sex],
    "cp": [cp],
    "trestbps": [trestbps],
    "chol": [chol],
    "fbs": [fbs_value],
    "restecg": [restecg_value],
    "thalch": [thalch],
    "exang": [exang_value],
    "oldpeak": [oldpeak],
    "slope": [slope_value],
    "ca": [ca_value],
    "thal": [thal_value]
})

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Heart Disease Risk",
    use_container_width=True
)

if predict_button:

    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0][1]
    probability_percent = probability * 100

    if prediction == 1:
        st.error(
            f"⚠️ Higher Predicted Risk\n\n"
            f"Estimated probability: **{probability_percent:.1f}%**"
        )
    else:
        st.success(
            f"✅ Lower Predicted Risk\n\n"
            f"Estimated probability: **{probability_percent:.1f}%**"
        )

    st.progress(float(probability))

    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ <b>Important:</b> CardioSense is an educational machine-learning
        project and is not a medical diagnostic tool. Predictions should not
        replace evaluation by a qualified healthcare professional.
        </div>
        """,
        unsafe_allow_html=True
    )