# scripts/train_model.py

import os
import joblib
from tslearn.datasets import UCR_UEA_datasets
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# 1. Setup folder
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------
# 2. Load dataset Epilepsy dari tslearn
# ------------------------------
dataset = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = dataset.load_dataset("Epilepsy")

print("Shape X_train:", X_train.shape)
print("Shape X_test :", X_test.shape)

# ------------------------------
# 3. Flatten time-series menjadi 1D per sampel
# Bentuk 3D -> 2D (samples, timesteps*channels)
# ------------------------------
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat  = X_test.reshape(X_test.shape[0], -1)
print("Shape setelah flatten:", X_train_flat.shape, X_test_flat.shape)

# ------------------------------
# 4. Label Encoding
# ------------------------------
label_enc = LabelEncoder()
y_train_enc = label_enc.fit_transform(y_train)
y_test_enc  = label_enc.transform(y_test)

# Simpan label encoder
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
joblib.dump(label_enc, encoder_path)
print("Label encoder disimpan ->", encoder_path)

# ------------------------------
# 5. Normalisasi fitur
# ------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flat)
X_test_scaled  = scaler.transform(X_test_flat)

# Simpan scaler
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print("Scaler disimpan ->", scaler_path)

# ------------------------------
# 6. Training SVM
# ------------------------------
svm_clf = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
svm_clf.fit(X_train_scaled, y_train_enc)

# Simpan model
model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
joblib.dump(svm_clf, model_path)
print("Model SVM disimpan ->", model_path)

# ------------------------------
# 7. Evaluasi
# ------------------------------
y_pred = svm_clf.predict(X_test_scaled)
acc = accuracy_score(y_test_enc, y_pred)
print("\n===== RESULT =====")
print("Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test_enc, y_pred))
