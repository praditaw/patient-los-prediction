# Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import phik
from PIL import Image
import matplotlib.pyplot as plt

def run():
    # Membuat title
    st.title('Aplikasi Prediksi Lama Tinggal Pasien (Length of Stay) di Rumah Sakit')

    st.write('Dibuat oleh **Pradita Ajeng Wiguna - RMT 042**')

    # Menambahkan gambar
    image = Image.open('./src/image.jpeg')
    st.image(image, caption='Pasien di Rumah Sakit')

    # Menampilkan DataFrame
    st.write('## Dataset')
    st.markdown("""Dataset ini berisi data historis pasien yang dirawat di rumah sakit, termasuk usia, jenis kelamin, dan lama tinggal di rumah sakit.
                Dataset ini akan digunakan untuk menganalisis faktor-faktor yang mempengaruhi lama tinggal pasien di rumah sakit.
                """)
    df = pd.read_csv('./src/P1M2_pradita_ajeng_dataset.csv')
    st.dataframe(df)

    # Membuat Header
    st.header('Exploratory Data Analysis (EDA)')

    # Distribusi Variabel Target 
    st.subheader("Distribusi Variabel Target: `length_of_stay_avg`")
    fig_target, ax_target = plt.subplots(1, 2, figsize=(12, 6))
    sns.histplot(df['length_of_stay_avg'], bins=30, kde=True, ax=ax_target[0], color='skyblue')
    ax_target[0].set_title('Distribusi Rata-Rata Lama Tinggal (length_of_stay_avg)')
    
    sns.boxplot(y=df['length_of_stay_avg'], ax=ax_target[1], color='lightcoral')
    ax_target[1].set_title('Boxplot Rata-Rata Lama Tinggal')
    st.pyplot(fig_target)
    st.write(f"Statistik Deskriptif untuk **`length_of_stay_avg`**:")
    st.write(df['length_of_stay_avg'].describe())
    st.markdown("""
        Dari visualisasi dan angka-angka ini, didapatkan beberapa *insight* penting mengenai rata-rata lama tinggal pasien di rumah sakit:
        1. Histogram menunjukkan data `length_of_stay_avg` terdistribusi cukup merata dan mendekati bentuk lonceng, dengan puncak frekuensi di sekitar 5 hari. Ini juga dikonfirmasi oleh nilai rata-rata (`mean` 4.99 hari) dan `median` (50% 4.98 hari) yang hampir sama.
        2. Boxplot dan kuartil menunjukkan bahwa sebagian besar pasien (50% di antaranya) memiliki rata-rata lama tinggal antara sekitar 4 hari (25% 3.96) hingga 6 hari (75% 6.00).
        3. Rentang antar kuartil (IQR = Q3 - Q1 = 6.01 - 3.96 ≈ 2.05 hari) memberi gambaran bahwa mayoritas data berada dalam rentang tersebut.
        4. Data ini menunjukkan lama tinggal pasien berkisar antara minimal 1 hari hingga maksimal 10 hari.
        5. Ada beberapa catatan pasien dengan lama tinggal sedikit lebih tinggi (mendekati 9-10 hari) yang terlihat sebagai titik di atas boxplot, namun jumlahnya tidak signifikan.
        """)
    st.markdown("---")
    
    # Distribusi Fitur Numerik Lainnya
    st.subheader("Distribusi Fitur Numerik Lainnya")
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    numerical_features.remove('length_of_stay_avg') # Hapus target dari list ini

    # Pengguna dapat memilih fitur numerik
    selected_num = st.selectbox("Pilih Fitur Numerik:", options=numerical_features)
    if selected_num:
            fig_num, ax_num = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(df[selected_num], bins=30, ax=ax_num[0], color='skyblue')
            ax_num[0].set_title(f'Distribusi {selected_num}')

            sns.boxplot(y=df[selected_num], ax=ax_num[1], color='lightcoral')
            ax_num[1].set_title(f'Boxplot {selected_num}')
            st.pyplot(fig_num)
            st.write(f"Statistik Deskriptif untuk **{selected_num}**:")
            st.write(df[selected_num].describe())
            # Inisialisasi interpretasi_teks di sini
            interpretasi_teks = f"Berikut adalah interpretasi untuk distribusi Fitur **{selected_num}**:\n"
            # Menambahkan interpretasi spesifik per fitur
            if selected_num == 'admission_count':
                 interpretasi_teks += """\n* Rata-rata (mean) sekitar 2.0 dan median (50%) juga 2.0. Sebaran (std) 1.4, dengan nilai dari 0 hingga 10.
                                 \n* Histogramnya menunjukkan puncak di sekitar 1-2, dan cenderung miring ke kanan (right-skewed), ekornya lebih panjang ke arah nilai tinggi. Boxplot juga menunjukkan adanya beberapa outlier di nilai yang lebih tinggi (di atas sekitar 6).
                 """
            elif selected_num == 'readmission_count':
                 interpretasi_teks += """\n* Rata-rata (mean) sangat rendah, sekitar 0.5, dan median (50%) adalah 0.0. Ini berarti lebih dari separuh waktu, tidak ada readmisi yang tercatat dalam satu jam. Nilai maksimalnya 5.0.
                                 \n* Histogramnya sangat miring ke kanan (right-skewed), dengan mayoritas data menumpuk di angka 0. Boxplotnya akan terlihat di bawah dengan beberapa outlier di atas.
                """
            elif selected_num == 'comorbid_conditions_count':
                 interpretasi_teks += """\n* Rata-rata (mean) dan median (50%) sama-sama 2.0. Rentangnya dari 0 hingga 11.0.
                                \n* Mirip dengan admission_count, distribusinya miring ke kanan (right-skewed) dengan puncak di sekitar 1-2 dan boxplotnya menunjukkan adanya outlier di sisi atas.
                """
            elif selected_num == 'daily_medication_dosage':
                 interpretasi_teks += """\n* Rata-rata (mean) dan median (50%) sangat dekat, sekitar 20.0. Standar deviasi (std) sekitar 5.0, dengan rentang nilai dari 5.0 hingga 40.0.
                                \n* Histogramnya menunjukkan fitur yang distribusinya paling mendekati simetris atau bentuk lonceng (normal) dibandingkan fitur numerik lainnya. Boxplotnya juga akan terlihat lebih seimbang dengan sedikit outlier di kedua ujungnya.
                """
            elif selected_num == 'emergency_visit_count':
                 interpretasi_teks += """\n* Rata-rata (mean) dan median (50%) adalah 1.0. Rentangnya dari 0 hingga 7.0.
                                \n* Distribusinya juga miring ke kanan (right-skewed), dengan banyak observasi di angka 0 dan 1, dan boxplotnya menunjukkan outlier di sisi atas.
                """
            else:
                interpretasi_teks += "\n* Belum ada interpretasi spesifik untuk fitur ini. "
            # Menampilkan interpretasi
            st.markdown(interpretasi_teks)

    st.markdown("---")

    # Hubungan Fitur Numerik dengan Target
    st.subheader("Hubungan Fitur Numerik dengan `length_of_stay_avg`")
    numerical_features_corr = df.select_dtypes(include=np.number).columns.tolist()
    numerical_features_corr.remove('length_of_stay_avg')
    # Pengguna dapat memilih fitur numerik
    selected_num_corr= st.selectbox("Pilih Fitur Numerik untuk Korelasi dengan Target:", options=numerical_features_corr, key="num_corr_target")
    if selected_num_corr:
            fig_corr_num, ax_corr_num = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=df[selected_num_corr], y=df['length_of_stay_avg'], ax=ax_corr_num, alpha=0.5, color='#4073FF')
            ax_corr_num.set_title(f'Scatter Plot: {selected_num_corr} vs. LoS')
            st.pyplot(fig_corr_num)
            correlation = df[selected_num_corr].corr(df['length_of_stay_avg'])
            st.write(f"Korelasi Pearson antara **{selected_num_corr}** dan **length_of_stay_avg**: **{correlation:.4f}**")

            # Inisialisasi interpretasi_teks di sini
            interpretasi_teks = f"Berikut adalah interpretasi untuk hubungan antara **{selected_num_corr}** dan **length_of_stay_avg**:\n"
            # Menambahkan interpretasi spesifik per fitur
            if selected_num_corr == 'admission_count':
                 interpretasi_teks += """\n* Titik-titik data tampak tersebar cukup merata secara vertikal di setiap nilai `admission_count`. 
                 Tidak terlihat adanya pola linear yang jelas (naik atau turun) antara jumlah penerimaan per jam dengan rata-rata lama tinggal.
                 Jumlah penerimaan dalam satu jam tertentu tampaknya tidak memiliki hubungan linear yang kuat dengan rata-rata lama tinggal pasien yang masuk pada jam tersebut.
                 """
            elif selected_num_corr == 'readmission_count':
                 interpretasi_teks += """\n* Mirip dengan `admission_count`, titik-titik tersebar secara vertikal di setiap nilai `readmission_count`. 
                 Tidak ada tren linear yang jelas.
                 Jumlah pasien yang diterima kembali dalam satu jam juga tampaknya tidak memiliki hubungan linear yang kuat dengan rata-rata lama tinggal.
                 """
            elif selected_num_corr == 'comorbid_conditions_count':
                 interpretasi_teks += """\n* Polanya serupa `admission_count`. Untuk setiap jumlah penyakit penyerta, sebaran `length_of_stay_avg` tampak mencakup hampir seluruh rentang (1-10 hari). Tidak ada tren linear yang jelas.
                 Jumlah penyakit penyerta, berdasarkan plot ini, tidak menunjukkan hubungan linear yang kuat dengan rata-rata lama tinggal. 
                 Mungkin ada hubungan non-linear atau interaksi dengan fitur lain, tapi secara visual linearitasnya lemah.
                 """
            elif selected_num_corr == 'daily_medication_dosage':
                 interpretasi_teks += """\n* Plot yang paling 'padat' dan berbeda. Titik-titik membentuk awan yang lebih terkonsentrasi di tengah, 
                 namun tidak ada pola linear yang jelas (garis lurus naik atau turun). Dosis obat harian tampaknya tidak memiliki hubungan linear yang kuat dengan rata-rata lama tinggal. 
                 """
            elif selected_num_corr == 'emergency_visit_count':
                 interpretasi_teks += """\n* Pola mirip dengan `admission_count` dan `readmission_count`. Titik-titik tersebar vertikal. 
                 Jumlah kunjungan darurat per jam juga tidak menunjukkan hubungan linear yang kuat dengan rata-rata lama tinggal. 
                 """
            else:
                interpretasi_teks += "\n* Belum ada interpretasi spesifik untuk fitur ini."
            # Menampilkan interpretasi
            st.markdown(interpretasi_teks)
    st.markdown("---")

    # Analisis Hubungan Semua Fitur dengan Target
    # Menghitung korelasi menggunakan phik
    # Definisikan kolom
    st.subheader("Analisis Hubungan Semua Fitur dengan `length_of_stay_avg`")
    nums = ['admission_count', 'readmission_count', 
            'comorbid_conditions_count', 'daily_medication_dosage', 
            'emergency_visit_count']
    
    cats = ['hospital_name', 'condition_type', 'patient_age_group', 
            'patient_gender', 'severity_level', 'seasonal_indicator', 
            'primary_diagnosis_code']
    
    # Konversi admission_date ke datetime
    df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
    
    # Ekstraksi fitur waktu
    df['admission_year'] = df['admission_date'].dt.year
    df['admission_month'] = df['admission_date'].dt.month_name()
    df['admission_day_of_week'] = df['admission_date'].dt.day_name()
    
    time = ['admission_year', 'admission_month', 'admission_day_of_week']
    target = 'length_of_stay_avg'

    all_for_phik = nums + cats + time + [target]
    df_phik = df[all_for_phik].copy()

    # Korelasi Phik
    phik_corr = df_phik.phik_matrix(interval_cols=nums + [target])

    # Tampilkan heatmap
    st.write("#### Heatmap Korelasi Phik")
    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(phik_corr, annot=True, fmt=".2f", cmap="YlGn", linewidths=0.5, vmin=0, vmax=1, ax=ax)
    plt.title('Heatmap Korelasi Phik Antar Semua Fitur', fontsize=15)
    st.pyplot(fig)

    # Korelasi fitur dengan target
    st.write("#### Nilai Korelasi Phik dengan Target (`length_of_stay_avg`)")
    phik_with_target = phik_corr['length_of_stay_avg'].sort_values(ascending=False)
    st.dataframe(phik_with_target)
    st.markdown("""
                * Heatmap secara keseluruhan didominasi **warna kuning muda**, yang menunjukkan nilai Phik sangat rendah (mendekati 0) di hampir semua pasangan fitur. 
                Ini berarti, **bahkan ketika memperhitungkan hubungan non-linear, hubungan antar fitur (termasuk dengan `length_of_stay_avg`) umumnya sangat lemah**. 
                * Fitur seperti `severity_level`, `patient_gender`, dan `daily_medication_dosage` menunjukkan nilai Phik tertinggi dengan `length_of_stay_avg`, namun nilainya masih sangat kecil (sekitar 0.01-0.02). 
                Sejumlah besar fitur lain (seperti `admission_count`, `hospital_name`, `condition_type`, `primary_diagnosis_code`, dll.) memiliki korelasi Phik praktis nol dengan `length_of_stay_avg`. 
                * Analisis Phik mengonfirmasi dan memperkuat temuan sebelumnya bahwa fitur-fitur pada dataset ini, bahkan ketika dianalisis untuk hubungan non-linear, 
                menunjukkan hubungan individual yang sangat lemah dengan `length_of_stay_avg`. **Tidak ada satu pun fitur yang menonjol sebagai prediktor kuat secara tunggal**.
                * Ini menunjukkan bahwa **memprediksi `length_of_stay_avg` dengan akurasi tinggi menggunakan fitur-fitur saat ini mungkin akan menantang**. 
                Keberhasilan model akan sangat bergantung pada kemampuannya menemukan pola dari interaksi antar beberapa fitur atau hubungan non-linear yang lebih kompleks, 
                yang mungkin tidak terdeteksi oleh analisis korelasi.
                """)

    st.markdown("---")

if __name__ == '__main__':
    run()