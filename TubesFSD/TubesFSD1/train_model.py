import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# 1. Memuat Dataset Wisconsin
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
feature_names = cancer.feature_names

# 2. Pembagian Data (70% Training, 30% Testing) dengan stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Scaling menggunakan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Melatih Model Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)

# 5. Menyimpan Model, Scaler, dan Nama Fitur ke file .pkl
joblib.dump(gnb_model, 'cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'features.pkl')

print("Selesai! File cancer_model.pkl, scaler.pkl, dan features.pkl telah dibuat.")