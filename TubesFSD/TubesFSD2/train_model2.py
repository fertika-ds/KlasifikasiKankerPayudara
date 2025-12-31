import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

# 1. Memuat Dataset dan Memilih 5 Fitur
cancer = load_breast_cancer()
selected_features = ['mean radius', 'mean texture', 'mean smoothness', 'mean concavity', 'mean symmetry']
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)[selected_features]
y = cancer.target

# 2. Bagi Data (70:30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. Pre-processing: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Melatih Model Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)

# 5. Menyimpan aset ke file .pkl
joblib.dump(gnb_model, 'model_5_features.pkl')
joblib.dump(scaler, 'scaler_5_features.pkl')
joblib.dump(selected_features, 'feature_names.pkl')

print("Berhasil! File pkl telah dibuat di folder TubesFSD.")