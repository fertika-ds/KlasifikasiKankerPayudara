import gradio as gr
import joblib
import numpy as np

# Load model, scaler, dan daftar fitur
model = joblib.load('model_5_features.pkl')
scaler = joblib.load('scaler_5_features.pkl')
features = joblib.load('feature_names.pkl')

def predict_cancer(radius, texture, smoothness, concavity, symmetry):
    # Susun input menjadi array
    input_data = np.array([[radius, texture, smoothness, concavity, symmetry]])
    # Scaling input
    input_scaled = scaler.transform(input_data)
    # Prediksi
    prediction = model.predict(input_scaled)
    
    return "Jinak (Benign)" if prediction[0] == 1 else "Ganas (Malignant)"

# Membuat UI dengan 5 input numerik
demo = gr.Interface(
    fn=predict_cancer,
    inputs=[gr.Number(label=f) for f in features],
    outputs="text",
    title="Diagnosis Kanker (5 Parameter)",
    description="Masukkan 5 nilai rata-rata karakteristik sel untuk diagnosis."
)

if __name__ == "__main__":
    demo.launch()