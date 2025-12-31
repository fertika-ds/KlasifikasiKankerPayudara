import streamlit as st
import joblib
import numpy as np

# Memuat aset model
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

st.title("🩺 Streamlit Cancer Diagnosis")
st.write("Prediksi klasifikasi sel tumor berdasarkan data biometrik.")

# Input menggunakan sidebar agar tampilan utama tetap bersih
st.sidebar.header("Input Fitur Sel")
user_inputs = []
for feature in features:
    val = st.sidebar.number_input(f"{feature}", value=0.0)
    user_inputs.append(val)

if st.button("Lakukan Prediksi"):
    # Preprocessing dan Prediksi
    input_array = np.array(user_inputs).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    
    # Menampilkan hasil dengan warna berbeda
    if prediction[0] == 0:
        st.error("Hasil Diagnosis: Ganas (Malignant)")
    else:
        st.success("Hasil Diagnosis: Jinak (Benign)")