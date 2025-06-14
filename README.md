# Data-Driven Hospital Management: Memprediksi Rata-Rata Lama Tinggal Pasien (LoS) Menggunakan Machine Learning

## Repository Outline

1.  `los_prediction.ipynb` - Notebook utama yang berisi seluruh proses analisis data, mulai dari pemuatan data, exploratory data analysis (EDA), feature engineering, pendefinisian model, pelatihan model, evaluasi, hingga penyimpanan model terbaik.
2.  `los_prediction_inference.ipynb` - Notebook yang digunakan untuk demonstrasi inferensi model. Notebook ini memuat model yang telah disimpan dan menggunakannya untuk melakukan prediksi pada data baru.
3.  `deployment/` - Folder yang berisi semua file yang dibutuhkan untuk deployment aplikasi web prediksi LoS (seperti `streamlit_app.py`, `requirements.txt`, `eda.py`, `prediction.py`, dan salinan model).
4.  `url.txt` - Berisi URL dataset yang digunakan dan URL aplikasi web hasil deployment.
5.  `README.md` - Dokumen yang menjelaskan gambaran umum proyek, tujuan, metodologi, dan hasil.

## Problem Background
Industri layanan kesehatan, khususnya rumah sakit, secara berkelanjutan menghadapi berbagai tantangan operasional yang kompleks. Beberapa isu yang sering muncul meliputi **peningkatan waktu tunggu pasien di Unit Gawat Darurat (UGD), optimalisasi penggunaan tempat tidur yang belum maksimal, dan tekanan pada biaya operasional termasuk biaya lembur staf.** Di sisi lain, pasien dan keluarga seringkali mengungkapkan kekhawatiran terkait **ketidakpastian durasi perawatan dan mengharapkan proses perencanaan kepulangan yang lebih proaktif dan jelas.** Rata-rata lama tinggal pasien (Length of Stay/LoS) diidentifikasi sebagai faktor kunci yang mempengaruhi berbagai aspek operasional tersebut. Oleh karena itu, kemampuan untuk memprediksi LoS secara tepat menjadi krusial untuk mengatasi masalah ini dan meningkatkan efisiensi serta kualitas layanan rumah sakit.<br>

**Objective**

Proyek ini bertujuan untuk **mengembangkan sebuah sistem prediktif menggunakan model machine learning untuk memprediksi rata-rata lama tinggal pasien** (`length_of_stay_avg`) menggunakan **algoritma KNN Regressor, SVR, Decesion Tree Regressor, Random Forest Regressor, dan Gradient Boosting Regressor** dengan metrik evaluasinya adalah **MAE dan $R^2$ Score**. Model ini diharapkan dapat memberikan insight berharga bagi tim manajemen untuk merancang langkah strategis dalam pengelolaan operasional RS.

## Project Output
Output utama dari proyek ini adalah:
1.  Sebuah **model machine learning (Gradient Boosting Regressor)** yang telah dilatih dan divalidasi untuk memprediksi rata-rata lama tinggal pasien (LoS) dalam satuan hari.
2.  Sebuah **aplikasi web interaktif (hasil deployment)** yang memungkinkan pengguna (seperti tim manajemen RS atau staf medis) untuk memasukkan data pasien dan mendapatkan prediksi LoS secara real-time.
3.  **Analisis dan insight** dari data historis pasien yang dapat digunakan untuk pengambilan keputusan strategis di rumah sakit.

## Data
Data yang digunakan dalam proyek ini merupakan dataset historis pasien dari Riyadh Hospital.
Nama Dataset    : Riyadh Hospital Admissions Dataset (2020–2024) <br>
URL Sumber      : https://www.kaggle.com/datasets/datasetengineer/riyadh-hospital-admissions-dataset-20202024 <br>

**Karakteristik Data**:
* Jumlah Baris: 41.544 baris
* Jumlah Kolom: 14 kolom
* Target Variabel: `length_of_stay_avg` (numerik, dalam satuan hari).
* Fitur: campuran fitur numerik (misalnya, `admission_count`, `daily_medication_dosage`) dan kategorikal (misalnya, `hospital_name`, `condition_type`, `patient_age_group`).
* Missing Values: setelah pengecekan awal, tidak ditemukan missing values pada dataset yang digunakan.
* Duplikasi Data: tidak ditemukan data duplikat.

## Method
Proyek ini menggunakan pendekatan **Supervised Learning** dengan fokus pada masalah **Regresi**, karena tujuannya adalah untuk memprediksi nilai kontinu yaitu rata-rata lama tinggal pasien (`length_of_stay_avg`).

Metodologi utama yang diterapkan meliputi:
1.  **Exploratory Data Analysis (EDA)**: untuk memahami distribusi data, mengidentifikasi pola, dan hubungan antar variabel.
2.  **Feature Engineering**: termasuk ekstraksi fitur dari tanggal (`admission_date`), penanganan outliers (menggunakan Winsorizer), encoding fitur kategorikal (One-Hot Encoding dan Ordinal Encoding), dan scaling fitur numerik (StandardScaler).
3.  **Model Definition & Training**: 
    * K-Nearest Neighbors (KNN) Regressor
    * Support Vector Regressor (SVR)
    * Decision Tree Regressor
    * Random Forest Regressor
    * Gradient Boosting Regressor (dipilih sebagai model terbaik)
4.  **Hyperparameter Tuning**: menggunakan RandomizedSearchCV pada model terbaik untuk optimasi performa.
5.  **Model Evaluation**: model dievaluasi menggunakan metrik Mean Absolute Error (MAE) dan R-squared ($R^2$) Score.
6.  **Preprocessing Pipeline**: Menggunakan `sklearn.pipeline.Pipeline` dan `sklearn.compose.ColumnTransformer` untuk menggabungkan semua langkah preprocessing dan model, memastikan konsistensi dan mencegah data leakage.
7.  **Model Deployment**: model terbaik dideploy sebagai aplikasi web menggunakan Streamlit dan Hugging Face Spaces.

Berikut tampilan dari Aplikasi Web - LoS Prediction
![EDA](<image1.png>)
![Prediction](<image2.png>)

## Stacks
* **Bahasa Pemrograman**: Python
* **Library Utama Python**:
    * `Pandas`: untuk manipulasi dan analisis data.
    * `NumPy`: untuk komputasi numerik.
    * `Scikit-learn`: untuk preprocessing, pembangunan model machine learning, evaluasi, dan pipeline.
    * `Matplotlib` & `Seaborn`: untuk visualisasi data.
    * `Pickle`: untuk menyimpan dan memuat model.
    * `Streamlit`: untuk membangun antarmuka aplikasi web.
    * `feature-engine`: untuk teknik feature engineering seperti Winsorizer.
    * `phik`: untuk analisis korelasi Phik.
* **Tools**:
    * Jupyter Notebook
    * Visual Studio Code
    * Git & GitHub
    * Hugging Face Spaces

## Reference
* **Dataset**: (https://www.kaggle.com/datasets/datasetengineer/riyadh-hospital-admissions-dataset-20202024)
* **Deployment URL**: https://huggingface.co/spaces/praditaw/los-prediction
* **Justifikasi Proyek**:
    * [Overcrowding in Emergency Department: Causes, Consequences, and Solutions—A Narrative Review](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6709765/)
    * [Why Reducing Length of Stay Is Critical for Patient & Hospital Well-being](https://www.definitivehc.com/blog/why-reducing-length-of-stay-is-critical-for-patient-hospital-well-being)
    * [Proactive discharge planning for more efficient patient flow: reduce “avoidable” days through actionable data](https://www.healthcatalyst.com/insights/proactive-discharge-planning-efficient-patient-flow)
    * [The Effect of Length of Hospital Stay and Patient Factors on Patient Satisfaction in an Academic Hospital](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6430029/)
* **Panduan Penulisan Markdown**:
    * [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)

## Kontributor
* [Pradita Ajeng Wiguna](https://github.com/praditaw)