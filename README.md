# 📚 Habit vs Exam — Prediksi Kelulusan Ujian Mahasiswa

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)

**Prediksi apakah seorang mahasiswa akan LULUS atau TIDAK LULUS ujian berdasarkan kebiasaan belajar dan gaya hidup mereka.**

</div>

---

## 🎯 Tentang Proyek

Proyek ini merupakan implementasi **Machine Learning** untuk memprediksi performa akademik mahasiswa berdasarkan kebiasaan sehari-hari mereka. Dengan memanfaatkan dataset *Student Habits vs Academic Performance* dari Kaggle, sistem ini mampu memprediksi apakah seorang mahasiswa akan **lulus (nilai ≥ 70)** atau **tidak lulus (nilai < 70)** dalam ujian.

> 💡 **Insight Utama**: Kebiasaan seperti jam belajar, kualitas tidur, tingkat kehadiran, dan kesehatan mental ternyata memiliki korelasi signifikan terhadap nilai ujian mahasiswa.

---

## ✨ Fitur Unggulan

| Fitur | Keterangan |
|-------|-----------|
| 🔮 **Prediksi Real-time** | Input data kebiasaan dan dapatkan prediksi instan lewat web app |
| 📊 **Analisis Mendalam** | Eksplorasi data, visualisasi korelasi, dan distribusi nilai ujian |
| 🤖 **5 Model ML** | Perbandingan Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, dan SVM |
| ⚙️ **Hyperparameter Tuning** | Optimasi model dengan GridSearchCV untuk akurasi terbaik |
| 📱 **UI Interaktif** | Antarmuka Streamlit yang ramah pengguna dengan form prediksi dinamis |
| 🔍 **Transparansi Data** | Tampilkan detail data yang telah diproses dan di-scale oleh model |

---

## 🗂️ Struktur Proyek

```
habit_vs_exam/
│
├── 📓 prediksi_nilai_ujian_berdasrkan_habit_uts_ml.ipynb   # Notebook analisis ML lengkap
├── 🌐 app.py                                               # Aplikasi web Streamlit
├── 🧠 habit_vs_exam_model.pkl                             # Model ML yang telah dilatih
└── 📋 requirements.txt                                    # Dependensi Python
```

---

## 📊 Dataset

- **Sumber**: [Student Habits vs Academic Performance — Kaggle](https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance)
- **Target**: `exam_score` → dikonversi ke **klasifikasi biner** (Lulus / Tidak Lulus)

### 🔤 Fitur Input (14 Variabel)

| No | Fitur | Deskripsi |
|----|-------|-----------|
| 1 | `age` | Usia mahasiswa |
| 2 | `gender` | Jenis kelamin |
| 3 | `study_hours_per_day` | Jam belajar per hari |
| 4 | `social_media_hours` | Jam penggunaan media sosial per hari |
| 5 | `netflix_hours` | Jam menonton Netflix per hari |
| 6 | `part_time_job` | Status kerja paruh waktu |
| 7 | `attendance_percentage` | Persentase kehadiran kuliah |
| 8 | `sleep_hours` | Jam tidur per hari |
| 9 | `diet_quality` | Kualitas pola makan |
| 10 | `exercise_frequency` | Frekuensi olahraga per minggu |
| 11 | `parental_education_level` | Tingkat pendidikan orang tua |
| 12 | `internet_quality` | Kualitas koneksi internet |
| 13 | `mental_health_rating` | Skor kesehatan mental (1–10) |
| 14 | `extracurricular_participation` | Keikutsertaan dalam kegiatan ekstrakurikuler |

---

## 🤖 Model Machine Learning

Lima algoritma klasifikasi diuji dan dibandingkan dalam proyek ini:

| Model | Keterangan |
|-------|-----------|
| **Logistic Regression** | Model baseline linear sederhana |
| **Decision Tree** | Pohon keputusan dengan splitting berbasis informasi |
| **Random Forest** | Ensemble dari banyak pohon keputusan |
| **Gradient Boosting** | Boosting iteratif untuk meningkatkan akurasi |
| **SVM** | Support Vector Machine dengan kernel optimal |

Model terbaik disimpan sebagai `habit_vs_exam_model.pkl` dan digunakan langsung oleh aplikasi Streamlit.

---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repositori
```bash
git clone https://github.com/nasrulaminmuis/habit_vs_exam.git
cd habit_vs_exam
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Web
```bash
streamlit run app.py
```

Buka browser dan akses: **http://localhost:8501**

---

## 🖥️ Tampilan Aplikasi

Setelah aplikasi berjalan, Anda akan melihat:

1. **Form Input** — Masukkan kebiasaan mahasiswa:
   - ⏰ Jam belajar per hari
   - 📱 Jam media sosial per hari
   - 🎬 Jam menonton Netflix per hari
   - 🏫 Persentase kehadiran kuliah
   - 😴 Jam tidur per hari
   - 🏃 Frekuensi olahraga (kali/minggu)
   - 🎓 Tingkat pendidikan orang tua
   - 🧠 Skor kesehatan mental (1–10)

2. **Hasil Prediksi** — Menampilkan:
   - ✅ **LULUS** (nilai ≥ 70) atau ❌ **TIDAK LULUS** (nilai < 70)
   - Probabilitas kelulusan dalam persentase
   - Detail data yang telah diproses oleh model

---

## 🛠️ Teknologi yang Digunakan

<div align="center">

| Teknologi | Versi | Kegunaan |
|-----------|-------|---------|
| ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white) | 3.8+ | Bahasa pemrograman utama |
| ![Streamlit](https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white) | Latest | Framework web app interaktif |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?logo=scikit-learn&logoColor=white) | Latest | Library machine learning |
| ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white) | Latest | Komputasi numerik |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white) | Latest | Manipulasi & analisis data |
| ![Joblib](https://img.shields.io/badge/-Joblib-gray) | Latest | Serialisasi model ML |

</div>

---

## 📈 Alur Kerja Proyek

```
📥 Dataset Kaggle
        ↓
🔍 Exploratory Data Analysis (EDA)
        ↓
🛠️ Preprocessing (Encoding, Scaling, Feature Selection)
        ↓
📊 Visualisasi (Distribusi, Korelasi, Scatter Plot)
        ↓
🤖 Training 5 Model Klasifikasi
        ↓
⚙️ Hyperparameter Tuning (GridSearchCV)
        ↓
💾 Simpan Model Terbaik (.pkl)
        ↓
🌐 Deploy ke Streamlit Web App
```

---

## 👥 Tim Pengembang

<div align="center">

**Kelompok Barokah Jaya**

| Nama | Peran |
|------|-------|
| 🧑‍💻 **Arifehan Maulana** | Machine Learning & Data Analysis |
| 🧑‍💻 **Nasrul Amin Muis** | Machine Learning & Web App Development |

</div>

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan **tugas akademik** dan bersifat open-source. Silakan gunakan dan kembangkan sesuai kebutuhan dengan mencantumkan atribusi yang sesuai.

---

<div align="center">

⭐ **Jika proyek ini bermanfaat, jangan lupa beri bintang!** ⭐

Made with ❤️ by **Kelompok Barokah Jaya**

</div>
