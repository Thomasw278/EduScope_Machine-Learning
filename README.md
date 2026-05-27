# EduScope - Machine Learning 🎓🤖

Repositori ini berisi alur kerja (*workflow*) dan pemodelan *Machine Learning* untuk menganalisis serta memprediksi performa kognitif dan perilaku psikologis siswa. Proyek ini memecah permasalahan menjadi dua pendekatan utama: **Regresi** untuk memprediksi nilai ujian secara kontinu, dan **Klasifikasi** untuk memprediksi tingkat motivasi siswa.

## 📌 Deskripsi Proyek
Faktor-faktor yang memengaruhi kesuksesan akademik sangatlah beragam, mulai dari jam belajar, tingkat kehadiran, hingga keterlibatan orang tua. Proyek ini bertujuan untuk mengekstrak *insight* dari data tersebut dan membangun model prediktif yang dapat membantu tenaga pendidik dalam memberikan intervensi yang tepat sasaran.

### Tujuan Utama:
1. **Memprediksi Capaian Akademik (`Exam_Score`)**: Menggunakan *Linear Regression* untuk memprediksi nilai akhir siswa (skala 0-100).
2. **Mengklasifikasi Tingkat Motivasi (`Motivation_Level`)**: Menggunakan *Logistic Regression* dan *Random Forest* untuk memetakan dorongan psikologis siswa ke dalam kategori *Low*, *Medium*, dan *High*.

---

## 📊 Informasi Dataset
Dataset yang digunakan berasal dari Kaggle: [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors/data).

* **Jumlah Fitur**: 18 Fitur Independen (seperti `Hours_Studied`, `Attendance`, `Parental_Involvement`, dll.)
* **Penanganan Data**: 
  * Imputasi *missing values* menggunakan nilai Modus.
  * Pembersihan *outliers* menggunakan metode *Interquartile Range* (IQR).
  * Transformasi data kategorial menggunakan `OrdinalEncoder` (untuk data bertingkat) dan `One-Hot Encoding` (untuk data nominal).

---

## 🧠 Pendekatan Pemodelan & Hasil Evaluasi

### 1. Prediksi Nilai Ujian (Regresi)
* **Algoritma**: Linear Regression
* **Fitur Utama**: `Hours_Studied`, `Attendance`, `Previous_Scores`, `Parental_Involvement`, `Access_to_Resources`
* **Metrik Evaluasi**: Menggunakan **R² Score** dan **Mean Absolute Error (MAE)**. Model memiliki hubungan linier yang kuat untuk memprediksi capaian akademik kognitif siswa.

### 2. Klasifikasi Tingkat Motivasi (Klasifikasi)
* **Algoritma**: Logistic Regression & Random Forest Classifier
* **Insight Penting (Limitasi Prediktif)**: 
  * Analisis *Feature Importance* dari *Random Forest* menunjukkan fitur `Previous_Scores` mendominasi prediksi hingga **~68%**, sementara fitur lainnya kurang informatif.
  * **Kesimpulan Analitis**: Akurasi klasifikasi yang berada di kisaran 38% bukanlah indikasi kegagalan algoritma, melainkan fakta objektif bahwa motivasi belajar (kondisi psikologis) bersifat non-linier, kompleks, dan sulit diprediksi hanya menggunakan fitur terstruktur pada dataset ini.

---

## 📂 Struktur Repositori

```text
├── Model Final.ipynb                # Notebook utama berisi EDA, Pre-processing, dan Training
├── StudentPerformanceFactors.csv    # Dataset mentah 
├── model_linear_exam.pkl            # Model diekspor: Linear Regression
├── model_logistic_motivation.pkl    # Model diekspor: Logistic Regression
├── scaler_model.pkl                 # StandardScaler untuk normalisasi data
├── image_e99e04.png                 # Aset gambar visualisasi
└── README.md                        # Dokumentasi proyek
