import streamlit as st
import numpy as np
import pickle
import os 
import base64

# Fungsi untuk mengatur latar belakang
def set_background(image_path="bg1.jpg"):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url('data:image/jpeg;base64,{encoded_image}');
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.5); /* Overlay untuk meningkatkan keterbacaan */
            z-index: -1;
        }}
        </style>
        """, unsafe_allow_html=True)
    else:
        st.warning("Background image not found. Default background will be used.")
        st.markdown("""
        <style>
        .stApp {
            background: #f0f2f6;
        }
        </style>
        """, unsafe_allow_html=True)

# Atur halaman (hanya sekali)
st.set_page_config(page_title="Diabetes Prediction Dashboard", layout="wide")

# Set background
set_background("bg1.jpg")

# Load model dan scaler dengan penanganan error
try:
    with open("model_svm.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model atau scaler tidak ditemukan. Silakan periksa file!")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model/scaler: {str(e)}")
    st.stop()

# Judul aplikasi
st.markdown("""
<div style="text-align: center;">
    <h1 style="color: #080808;">Prediksi Dini Diabetes</h1>
    <p>Aplikasi prediksi risiko diabetes.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar menu
st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Navigasi", ["Dashboard", "Tentang", "Kelompok 8"], label_visibility="collapsed")

if menu == "Dashboard":
    st.subheader("Masukkan Data Anda")

    # Tata letak input dengan dua kolom
    col1, col2 = st.columns(2)
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, help="Jumlah kehamilan yang pernah dialami.")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, help="Kadar glukosa darah setelah puasa.")
        blood_pressure = st.number_input("BloodPressure (mmHg)", min_value=0, max_value=200, help="Tekanan darah diastolik.")
        skin_thickness = st.number_input("SkinThickness (mm)", min_value=0, max_value=100, help="Ketebalan lipatan kulit.")
    with col2:
        insulin = st.number_input("Insulin (μU/mL)", min_value=0, max_value=900, help="Kadar insulin dalam darah.")
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, help="Indeks Massa Tubuh (kg/m²).")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, help="Faktor keturunan diabetes.")
        age = st.number_input("Age", min_value=0, max_value=120, help="Usia pasien.")

    if st.button("Dashboard"):
        # Validasi input
        if any([pregnancies < 0, glucose < 0, blood_pressure < 0, skin_thickness < 0, 
                insulin < 0, bmi < 0, dpf < 0, age < 0]):
            st.error("Input tidak boleh negatif!")
        else:
            try:
                input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                        insulin, bmi, dpf, age]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]

                # Tampilkan probabilitas jika tersedia
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_scaled)[0]
                    st.write(f"Probabilitas Diabetes: {prob[1]:.2%}")

                if prediction == 1:
                    st.error("Hasil Prediksi: Risiko Diabetes Terdeteksi")
                else:
                    st.success("Hasil Prediksi: Tidak Terindikasi Diabetes")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

elif menu == "Tentang":
    st.subheader("Tentang Aplikasi")
    st.markdown("""
Aplikasi ini dirancang untuk membantu prediksi dini diabetes berdasarkan delapan parameter medis penting, seperti kadar glukosa, tekanan darah, indeks massa tubuh (BMI), dan usia. Dengan hanya memasukkan data sederhana, pengguna dapat langsung mengetahui risiko diabetes secara cepat dan mudah. Deteksi dini ini sangat penting karena memungkinkan penanganan lebih awal guna mencegah komplikasi serius di kemudian hari. Sebagai alat bantu yang praktis bagi individu maupun tenaga kesehatan dalam melakukan screening awal, aplikasi ini memanfaatkan algoritma Support Vector Machine (SVM), salah satu model machine learning terbaik untuk klasifikasi medis. SVM bekerja dengan mencari garis pemisah optimal antara data sehat dan data berisiko, berdasarkan pola dari data sebelumnya, sehingga menghasilkan prediksi yang akurat dan andal.

Dataset: **Diabetes Prediction Dataset** (Kaggle)

Fitur input:
- Pregnancies: Jumlah kehamilan
- Glucose: Kadar glukosa darah
- BloodPressure: Tekanan darah diastolik
- SkinThickness: Ketebalan lipatan kulit
- Insulin: Kadar insulin
- BMI: Indeks Massa Tubuh
- DiabetesPedigreeFunction: Faktor keturunan diabetes
- Age: Usia pasien
""")

elif menu == "Kelompok 8":
    st.subheader("Profil Kelompok")
    st.markdown("""
**Mata Kuliah:** Pembelajaran Mesin Dasar 2023C 
**Dosen Pengampu:**  
- Dr. Elly Matul Imah, M.Kom  
- Yuni Rosita Dewi, S.Si., M.Si

**Kelompok 8:**  
- Marshanda Claudia Iswahyono (23031554014)  
- Thea Bayu Revalina (23031554035)  
- Hannia Harry Putri (23031554077)  
- Bintang Prananda Putra (23031554131)
""")

# Footer
st.markdown("""
<hr style="margin-top: 50px;">
<div style="text-align: center; font-size: 0.9em;">
    <p>© 2025 - Sistem Prediksi Dini Diabetes | Kelompok 8</p>
</div>
""", unsafe_allow_html=True)