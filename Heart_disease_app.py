import streamlit as st
import numpy as np
import pickle

# Page Configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="🏥", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        background-color: #e74c3c;
        color: white;
        width: 100%;
        padding: 10px;
        font-size: 18px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        color: white;
    }
    .title {
        text-align: center;
        color: #e74c3c;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    with open('model/rf_model_heart_disease.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header
st.markdown('<p class="title">🏥 Heart Disease Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient medical details to predict heart disease risk</p>', unsafe_allow_html=True)
st.markdown("---")

# Input Fields
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age      = st.number_input("Age", min_value=20, max_value=100, value=45)
    sex      = st.selectbox("Sex", ["Male", "Female"])
    cp       = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
    chol     = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fbs      = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg  = st.selectbox("Resting ECG", [0, 1, 2])

with col2:
    thalach  = st.number_input("Max Heart Rate", min_value=60, max_value=220, value=150)
    exang    = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak  = st.number_input("ST Depression", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    slope    = st.selectbox("Slope", [0, 1, 2])
    ca       = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal     = st.selectbox("Thalassemia", [0, 1, 2, 3])

st.markdown("---")

# Predict Button
if st.button("Predict Now"):

    sex_val = 1 if sex == "Male" else 0

    input_data = np.array([[age, sex_val, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction  = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Heart Disease Detected!")
        st.markdown(f"### Probability of Heart Disease: **{probability[1]*100:.1f}%**")
        st.progress(int(probability[1]*100))
        st.warning("Please consult a cardiologist immediately.")
    else:
        st.success("No Heart Disease Detected!")
        st.markdown(f"### Healthy Probability: **{probability[0]*100:.1f}%**")
        st.progress(int(probability[0]*100))
        st.info("Keep maintaining a healthy lifestyle!")

    # Summary Table
    st.markdown("---")
    st.subheader("Patient Summary")
    summary = {
        "Age": age, "Sex": sex, "Chest Pain": cp,
        "Blood Pressure": trestbps, "Cholesterol": chol,
        "Max Heart Rate": thalach
    }
    st.table(summary)

    st.caption("This is a ML prediction only. Always consult a doctor.")

# Sidebar
with st.sidebar:
    st.header("About")
    st.success("Model: Random Forest")
    st.info("Accuracy: 88.52%")
    st.markdown("---")
    st.markdown("**Models Compared:**")
    st.markdown("- Logistic Regression: 86.89%")
    st.markdown("- Decision Tree: 78.69%")
    st.markdown("- Random Forest: 88.52%")