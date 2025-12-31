import streamlit as st
import joblib
import numpy as np

# Load assets
model = joblib.load('model_5_features.pkl')
scaler = joblib.load('scaler_5_features.pkl')
features = joblib.load('feature_names.pkl')

st.set_page_config(page_title="Cancer Detection", page_icon="🎗️")

st.title("🎗️ Klasifikasi Kanker Payudara")
st.write("Prediksi berdasarkan 5 fitur utama: Radius, Texture, Smoothness, Concavity, dan Symmetry.")

# Input Form
st.sidebar.header("Parameter Input")
inputs = []
for f in features:
    val = st.sidebar.number_input(f"Input {f}", value=0.0, format="%.4f")
    inputs.append(val)

if st.button("Analisis"):
    # Reshape dan Scaling
    data = np.array([inputs])
    data_scaled = scaler.transform(data)
    
    # Prediksi & Probabilitas
    pred = model.predict(data_scaled)
    prob = model.predict_proba(data_scaled)

    st.subheader("Hasil Diagnosis:")
    if pred[0] == 0:
        st.error(f"**GANAS (Malignant)**")
    else:
        st.success(f"**JINAK (Benign)**")
    
    # Tampilan Probabilitas
    st.write(f"Keyakinan Model: {np.max(prob)*100:.2f}%")