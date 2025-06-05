# UAS Pembelajaran Mesin Dasar: Prediksi dini diabetes berdasarkan data klinis dan demografi

## Proyek
**Judul**: Prediksi dini diabetes berdasarkan data klinis dan demografi menggunakan algoritma Support Vector Machine (SVM)  
**Tujuan**: Proyek ini bertujuan untuk memprediksi dini diabetes berdasarkan 8 indikator yang berhubungan dengan kesehatan menggunakan machine learning SVM

**Anggota Tim (Kelompok 8 - 2023C)**:  
1. Marshanda Claudia Iswahyono (23031554014)
2. Thea Bayu Revalina          (23031554035)
3. Hannia Harry Putri          (23031554077)
4. Bintang Prananda Putra      (23031554131)


## Struktur Folder
```
📁 PMD_UAS/
├── app.py # Aplikasi utama Streamlit
├── bg1.jpg # Background image untuk tampilan UI
├── dataset_diabetes.csv # Dataset asli
├── diabetes_cleaning.csv # Dataset yang telah dibersihkan
├── model_svm.pkl # File model SVM terlatih
├── scaler.pkl # File scaler (normalisasi data)
├── svm_manual.py # Implementasi SVM manual (tanpa scikit-learn)
├── svm_rbf.py # Implementasi SVM dengan kernel RBF
├── requirements.txt # Daftar pustaka Python yang dibutuhkan

```

##  Cara Menjalankan Streamlit Secara Lokal

1. **Clone repository ini**:
   ```bash
   git clone https://github.com/username/diabetes-streamlit-app.git
   cd diabetes-streamlit-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**:
   ```bash
   streamlit run app.py
   ```
---

## Deploy ke Streamlit Community Cloud

1. Push semua file ke GitHub.
2. Buka [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Klik **“New app”**.
4. Pilih repository dan file `app.py`.
5. Klik **Deploy** — aplikasi akan online dalam hitungan detik.

---

## Fitur Aplikasi

- Input data kesehatan pengguna:
  - Glucose
  - Blood Pressure
  - BMI
  - Insulin
  - Age
  - dll.
- Prediksi risiko diabetes berdasarkan input.
- Model menggunakan SVM yang telah dilatih dan disimpan dalam `model_svm.pkl`.
- Tampilan interaktif dan mudah digunakan.

---

## Dataset

Dataset yang digunakan berasal dari:
- Pima Indians Diabetes Database (Kaggle)
- Disimpan dalam `dataset_diabetes.csv`
- Telah dibersihkan menjadi `diabetes_cleaning.csv`

---

## Library yang Digunakan

- `streamlit`
- `pandas`
- `scikit-learn`
- `joblib`
- `numpy`

---