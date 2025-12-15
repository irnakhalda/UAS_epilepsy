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
svm_clf  = joblib.load(MODEL_PATH)
scaler   = joblib.load(SCALER_PATH)
label_enc = joblib.load(ENCODER_PATH)

N_FEATURES = scaler.n_features_in_  # = 618

# ------------------------------
# 3. Streamlit UI
# ------------------------------
st.title("Prediksi Epilepsi (SVM)")

st.markdown("""
Masukkan data time-series Epilepsy  
- Upload CSV **tanpa header**
- Jumlah fitur **618 (206 × 3)**
- Kolom pertama **opsional label**
""")

# ------------------------------
# Fungsi prediksi
# ------------------------------
def predict_and_show(X_input):
    # Safety check
    if X_input.shape[0] == 0:
        st.error("Data kosong (0 sampel).")
        st.stop()

    # Scaling
    X_scaled = scaler.transform(X_input)

    # Prediksi
    y_pred_enc = svm_clf.predict(X_scaled)
    y_pred = label_enc.inverse_transform(y_pred_enc)

    # Probabilitas
    if hasattr(svm_clf, "predict_proba"):
        y_proba = svm_clf.predict_proba(X_scaled)
    else:
        df_dec = svm_clf.decision_function(X_scaled)
        y_proba = np.exp(df_dec) / np.sum(np.exp(df_dec), axis=1, keepdims=True)

    proba_df = pd.DataFrame(y_proba, columns=label_enc.classes_)

    # Statistik
    unique, counts = np.unique(y_pred, return_counts=True)
    stats_df = pd.DataFrame({"Kelas": unique, "Jumlah": counts})

    return y_pred, stats_df, proba_df

# ------------------------------
# Upload CSV (FIX UTAMA DI SINI)
# ------------------------------
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, header=None)

    # LOGIKA AMAN BERDASARKAN JUMLAH KOLOM
    if df.shape[1] == N_FEATURES + 1:
        # Ada label
        y_true = df.iloc[:, 0].astype(str)
        X_input = df.iloc[:, 1:].values

    elif df.shape[1] == N_FEATURES:
        # Tanpa label
        y_true = None
        X_input = df.values

    else:
        st.error(
            f"Jumlah kolom SALAH ({df.shape[1]}).\n"
            f"Harus {N_FEATURES} (tanpa label) atau {N_FEATURES + 1} (dengan label)."
        )
        st.stop()

    y_pred, stats_df, proba_df = predict_and_show(X_input)

    st.subheader("Hasil Prediksi")
    st.dataframe(pd.DataFrame(y_pred, columns=["Prediksi"]))

    st.subheader("Statistik Prediksi")
    st.dataframe(stats_df)

    fig, ax = plt.subplots()
    sns.barplot(data=stats_df, x="Kelas", y="Jumlah", ax=ax)
    st.pyplot(fig)

    st.subheader("Probabilitas Kelas")
    st.dataframe(proba_df.round(3))

    if y_true is not None:
        acc = np.mean(y_pred == y_true.values)
        st.success(f"Akurasi data upload: {acc:.2f}")

# ------------------------------
# Input Manual
# ------------------------------
st.subheader("Prediksi Manual (618 fitur)")

manual_input = st.text_area(
    "Masukkan 618 angka (pisahkan koma / newline)",
    height=200
)

if st.button("Prediksi Manual"):
    try:
        cleaned = manual_input.replace("\n", ",").replace(" ", "")
        values = [float(x) for x in cleaned.split(",") if x != ""]

        if len(values) != N_FEATURES:
            st.error(f"Jumlah fitur harus {N_FEATURES}, sekarang {len(values)}")
            st.stop()

        sample = np.array(values).reshape(1, -1)

        y_pred, stats_df, proba_df = predict_and_show(sample)

        st.success(f"Prediksi: {y_pred[0]}")
        st.dataframe(proba_df.round(3))

    except Exception as e:
        st.error(f"Error: {e}")
