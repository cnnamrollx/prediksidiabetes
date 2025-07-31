# Diabetes Prediction App

## Deskripsi
Aplikasi ini adalah framework Streamlit untuk memprediksi diabetes menggunakan model Random Forest yang dilatih dari dataset diabetes. Pengguna dapat memasukkan data secara manual atau mengunggah file CSV untuk prediksi batch.

## Persyaratan
Dependensi yang diperlukan tercantum di `requirements.txt`:
- streamlit==1.36.0
- scikit-learn==1.5.1
- pandas==2.2.2
- numpy==2.0.0
- joblib==1.4.2

## Instalasi
1. Pastikan Python terinstal (versi 3.6+).
2. Instal dependensi dengan perintah:
   ```
   pip install -r requirements.txt
   ```

## Cara Penggunaan
1. Jalankan skrip pelatihan untuk menghasilkan model (jika belum ada):
   ```
   python diabetes_random_forest.py
   ```
   Ini akan membuat `rf_diabetes_model.joblib`, `scaler.joblib`, dan `metrics.json`.

2. Jalankan aplikasi Streamlit:
   ```
   python -m streamlit run app.py
   ```
   Akses di browser melalui http://localhost:8501 atau http://192.168.1.9:8501

3. Di aplikasi:
   - Pilih mode: Input Manual atau Upload CSV.
   - Untuk Input Manual: Masukkan nilai fitur dan klik 'Predict'.
   - Untuk Upload CSV: Unggah file dengan kolom yang diperlukan, lihat hasil prediksi, dan unduh CSV.

## File Utama
- `diabetes_random_forest.py`: Skrip pelatihan model.
- `app.py`: Aplikasi Streamlit.
- `diabetes.csv`: Dataset contoh.

Untuk pertanyaan lebih lanjut, hubungi pengembang.