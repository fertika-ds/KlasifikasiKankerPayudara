# Klasifikasi Kanker Payudara (Breast Cancer Classification)

Proyek ini bertujuan untuk mengklasifikasikan kanker payudara sebagai **Ganas (Malignant)** atau **Jinak (Benign)** berdasarkan fitur-fitur yang diambil dari citra digital aspirasi jarum halus (FNA) dari massa payudara. Proyek ini menggunakan algoritma **Gaussian Naive Bayes** dan menyajikan model prediksi melalui antarmuka web interaktif menggunakan **Streamlit** dan **Gradio**.

## Deskripsi

Dataset yang digunakan adalah **Wisconsin Breast Cancer Dataset** yang tersedia di library `scikit-learn`. Proyek ini mengeksplorasi dua pendekatan pemodelan:
1.  **Model Full Features**: Menggunakan seluruh fitur yang tersedia dalam dataset.
2.  **Model 5 Features**: Menggunakan 5 fitur terpilih (*mean radius, mean texture, mean smoothness, mean concavity, mean symmetry*) untuk penyederhanaan namun tetap menjaga akurasi yang baik.

## Struktur Folder

```
KlasifikasiKankerPayudara/
├── TUBES_FSD.ipynb         # Notebook Jupyter untuk eksplorasi data dan eksperimen
├── TubesFSD/
│   ├── TubesFSD1/          # Pendekatan 1: Menggunakan seluruh fitur
│   │   ├── app_gradio.py   # Aplikasi web Gradio
│   │   ├── app_streamlit.py# Aplikasi web Streamlit
│   │   └── train_model.py  # Skript untuk melatih model
│   └── TubesFSD2/          # Pendekatan 2: Menggunakan 5 fitur terpilih
│       ├── app_gradio.py   # Aplikasi web Gradio
│       ├── app_streamlit.py# Aplikasi web Streamlit
│       └── train_model2.py # Skript untuk melatih model
└── README.md
```

## Instalasi

Pastikan Anda memiliki Python yang terinstal. Install library yang dibutuhkan dengan menjalankan perintah berikut:

```bash
pip install pandas scikit-learn joblib streamlit gradio notebook
```

## Penggunaan

Anda dapat menjalankan script pelatihan model terlebih dahulu untuk menghasilkan file model (`.pkl`), kemudian menjalankan aplikasi web.

### 1. Menjalankan Model Full Features (TubesFSD1)

**Latih Model:**
```bash
cd TubesFSD/TubesFSD1
python train_model.py
```
Ini akan menghasilkan file `cancer_model.pkl`, `scaler.pkl`, dan `features.pkl`.

**Jalankan Aplikasi Streamlit:**
```bash
streamlit run app_streamlit.py
```

**Jalankan Aplikasi Gradio:**
```bash
python app_gradio.py
```

### 2. Menjalankan Model 5 Fitur (TubesFSD2)

**Latih Model:**
```bash
cd TubesFSD/TubesFSD2
python train_model2.py
```
Ini akan menghasilkan file `model_5_features.pkl`, `scaler_5_features.pkl`, dan `feature_names.pkl`.

**Jalankan Aplikasi Streamlit:**
```bash
streamlit run app_streamlit.py
```

**Jalankan Aplikasi Gradio:**
```bash
python app_gradio.py
```

## Teknologi yang Digunakan

*   **Python**: Bahasa pemrograman utama.
*   **Scikit-learn**: Untuk pemuatan dataset, preprocessing, dan algoritma Machine Learning (Gaussian Naive Bayes).
*   **Pandas**: Untuk manipulasi data.
*   **Joblib**: Untuk menyimpan dan memuat model yang telah dilatih.
*   **Streamlit**: Framework untuk membuat aplikasi web data interaktif.
*   **Gradio**: Framework untuk membuat demo antarmuka machine learning.
