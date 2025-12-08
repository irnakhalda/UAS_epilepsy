# app.py

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ------------------------------
# 1. Setup folder model
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH   = os.path.join(MODEL_DIR, "svm_model.pkl")
SCALER_PATH  = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# ------------------------------
# 2. Load model, scaler, encoder
# ------------------------------
svm_clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_enc = joblib.load(ENCODER_PATH)

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.title("Prediksi Epilepsi (SVM)")

st.markdown("""
Masukkan data time-series Epilepsy.
- Bisa upload CSV atau input manual
- Format CSV: Kolom pertama (opsional) kelas, kolom selanjutnya fitur time-series
- Format manual: Pisahkan angka dengan koma atau newline
""")

# ------------------------------
# Fungsi prediksi dan statistik
# ------------------------------
def predict_and_show(X_input):
    # Flatten jika input 3D
    if len(X_input.shape) == 3:
        X_input = X_input.reshape(X_input.shape[0], -1)
    
    # Scaling
    X_input_scaled = scaler.transform(X_input)
    
    # Prediksi
    y_pred_enc = svm_clf.predict(X_input_scaled)
    y_pred = label_enc.inverse_transform(y_pred_enc)
    
    # Probabilitas
    if hasattr(svm_clf, "predict_proba"):
        y_proba = svm_clf.predict_proba(X_input_scaled)
        proba_df = pd.DataFrame(y_proba, columns=label_enc.classes_)
    else:
        df_dec = svm_clf.decision_function(X_input_scaled)
        y_proba = np.exp(df_dec) / np.sum(np.exp(df_dec), axis=1, keepdims=True)
        proba_df = pd.DataFrame(y_proba, columns=label_enc.classes_)
    
    # Statistik prediksi
    unique, counts = np.unique(y_pred, return_counts=True)
    stats_df = pd.DataFrame({"Kelas": unique, "Jumlah": counts})
    
    return y_pred, stats_df, proba_df

# ------------------------------
# Fungsi visualisasi
# ------------------------------
def plot_statistics(stats_df, title="Statistik Prediksi"):
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Kelas", y="Jumlah", data=stats_df, palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Jumlah Sampel")
    ax.set_xlabel("Kelas")
    st.pyplot(fig)

def plot_probabilities(proba_df, title="Probabilitas tiap kelas"):
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(proba_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Kelas")
    ax.set_ylabel("Sampel")
    st.pyplot(fig)

# ------------------------------
# Upload CSV
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Jika ada kolom label, drop
    if df.shape[1] > 1:
        X_input = df.iloc[:, 1:].values
    else:
        X_input = df.values

    y_pred, stats_df, proba_df = predict_and_show(X_input)
    
    # Tampilkan hasil
    st.subheader("Prediksi Kelas CSV:")
    st.dataframe(pd.DataFrame(y_pred, columns=["Prediksi"]))
    
    st.subheader("Statistik Prediksi CSV:")
    st.dataframe(stats_df)
    plot_statistics(stats_df, "Distribusi Prediksi CSV")
    
    st.subheader("Probabilitas tiap kelas (CSV):")
    st.dataframe(proba_df.round(3))
    plot_probabilities(proba_df, "Probabilitas tiap kelas CSV")
    
    # Jika ada label asli, tampilkan akurasi
    if df.shape[1] > 1:
        y_true = df.iloc[:, 0].astype(str)
        acc = np.mean(y_pred == y_true)
        st.write(f"Akurasi pada data upload: {acc:.2f}")

# ------------------------------
# Input manual
# ------------------------------
st.subheader("Prediksi Manual")

manual_input = st.text_area(
    "Masukkan deretan angka fitur time-series (pisahkan dengan koma atau newline)",
    height=150
)

if st.button("Prediksi Manual"):
    if manual_input.strip() != "":
        try:
            cleaned = manual_input.replace("\n", ",").replace("\t", ",").replace(" ", "")
            sample = np.array([float(x) for x in cleaned.split(",") if x != ""]).reshape(1, -1)
            
            y_pred, stats_df, proba_df = predict_and_show(sample)
            
            st.write("Prediksi Kelas Manual:", y_pred[0])
            
            st.subheader("Statistik Prediksi Manual:")
            st.dataframe(stats_df)
            plot_statistics(stats_df, "Distribusi Prediksi Manual")
            
            st.subheader("Probabilitas tiap kelas (Manual):")
            st.dataframe(proba_df.round(3))
            plot_probabilities(proba_df, "Probabilitas tiap kelas Manual")
            
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
    else:
        st.warning("Silakan masukkan data time-series sebelum prediksi.")
