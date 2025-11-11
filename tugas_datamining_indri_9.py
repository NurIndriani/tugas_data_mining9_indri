import streamlit as st
import pickle
import numpy as np
import Orange
from Orange.data import Table

# Judul aplikasi
st.title("🌸 Prediksi Kategori Bunga Iris")
st.write("Masukkan nilai fitur di bawah ini untuk memprediksi jenis bunga Iris.")

# Load model
model_path = "tugas_data_mining_9_indri.pkcls"  # nama file modelmu
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Input fitur dari pengguna
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
sepal_width  = st.number_input("Sepal Width (cm)",  min_value=0.0, max_value=10.0, value=3.5)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
petal_width  = st.number_input("Petal Width (cm)",  min_value=0.0, max_value=10.0, value=0.2)

# Prediksi saat tombol diklik
if st.button("🔍 Prediksi"):
    input_data = Table(model.domain, [[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model(input_data)
    st.subheader("Hasil Prediksi:")
    st.success(f"Model memprediksi bunga termasuk ke dalam kategori: **{prediction[0]}**")

st.markdown("---")
st.caption("Dibuat oleh Nur Indriani | UIN Alauddin Makassar")
