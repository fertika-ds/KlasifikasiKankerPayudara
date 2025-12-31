import gradio as gr
import joblib
import numpy as np

# Memuat model dan scaler yang telah disimpan
model = joblib.load('cancer_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

def predict_cancer(*args):
    # Mengonversi input menjadi array 2D
    data = np.array(args).reshape(1, -1)
    # Menstandarisasi input sesuai skala data training
    data_scaled = scaler.transform(data)
    # Melakukan prediksi
    prediction = model.predict(data_scaled)
    
    # Hasil diagnosis: 0 = Ganas, 1 = Jinak
    return "Jinak (Benign)" if prediction[0] == 1 else "Ganas (Malignant)"

# Membuat komponen input untuk ke-30 fitur
inputs = [gr.Number(label=name) for name in features]

demo = gr.Interface(
    fn=predict_cancer,
    inputs=inputs,
    outputs="text",
    title="Gradio: Diagnosis Kanker Payudara",
    description="Masukkan nilai biometrik untuk klasifikasi Ganas/Jinak menggunakan Gaussian NB."
)

if __name__ == "__main__":
    demo.launch()