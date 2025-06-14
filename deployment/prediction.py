#import libraries

import pickle
import pandas as pd
import streamlit as st

# Load model

with open('./src/best_los_gbt_model.pkl', 'rb') as file_1:
   model_pipeline = pickle.load(file_1)


def run ():
    st.title('🏥 Prediksi Rata-Rata Lama Tinggal Pasien (Length of Stay)')
    st.write("""
    Aplikasi ini memprediksi rata-rata lama tinggal pasien berdasarkan data yang Anda masukkan.
    """)
    # Pembuatan form input
    with st.form(key='los_prediction_form'):
        st.subheader("Masukkan Data Pasien:")
        # Input dari Pengguna 
        st.write("**Informasi Dasar & Riwayat**")
        admission_count = st.number_input('Jumlah Kunjungan Rawat Inap', min_value=0, value=0, step=1, key="ac_form")
        readmission_count = st.number_input('Jumlah Readmisi', min_value=0, value=0, step=1, key="rc_form")
        emergency_visit_count = st.number_input('Jumlah Kunjungan UGD', min_value=0, value=0, step=1, key="evc_form")
        comorbid_conditions_count = st.number_input('Jumlah Kondisi Komorbid', min_value=0, value=0, step=1, key="ccc_form")
        daily_medication_dosage = st.number_input('Dosis Obat Harian (mg)', min_value=0.0, value=5.0, step=0.1, format="%.1f", key="dmd_form")
        st.markdown('---')
        
        st.write("**Detail Admisi & Pasien**")
        hospital_name_options = ['Mecca City Hospital', 'Dammam General Hospital','Medina Specialist Hospital', 'Dammam Central Hospital', 
                                 'King Saud Hospital', 'Jeddah National Hospital', 'Riyadh National Hospital','Riyadh General Hospital', 'Other']
        hospital_name = st.selectbox('Nama Rumah Sakit', options=hospital_name_options, key="hn_form")
        condition_type_options = ['Asthma', 'COPD', 'Heart Attack', 'Other Respiratory Issues'] 
        condition_type = st.selectbox('Tipe Kondisi Medis Utama', options=condition_type_options, key="ct_form")
        patient_age_group_options = ['0-17', '18-45', '46-65', '66+'] 
        patient_age_group = st.selectbox('Kelompok Usia Pasien', options=patient_age_group_options, key="pag_form")
        patient_gender_options = ['Male', 'Female'] 
        patient_gender = st.selectbox('Jenis Kelamin Pasien', options=patient_gender_options, key="pg_form")
        severity_level_options = ['Mild', 'Moderate', 'Severe']
        severity_level = st.selectbox('Tingkat Keparahan Kondisi', options=severity_level_options, key="sl_form")
        st.markdown('---')

        st.write("**Informasi Tambahan**")
        seasonal_indicator_options = ['Winter', 'Spring', 'Summer', 'Fall']
        seasonal_indicator = st.selectbox('Indikator Musiman', options=seasonal_indicator_options, key="si_form")
        primary_diagnosis_code_options = ['Other', 'I21', 'J45', 'J44']
        primary_diagnosis_code = st.selectbox('Kode Diagnosis Utama', options=primary_diagnosis_code_options, key="pdc_form")
        admission_day_of_week = st.selectbox('Hari Masuk', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], 
                                              index=0, key="adow_form")
        admission_month = st.selectbox('Bulan Masuk', options=['January', 'February', 'March', 'April', 'May', 'June', 'July', 
                                                               'August', 'September', 'October', 'November', 'December'], index=0, key="amn_form")
        admission_year = st.number_input('Tahun Masuk', min_value=2020, max_value=2025, value=2024, step=1, key="ay_form")
        st.markdown('---')

        submitted = st.form_submit_button('Predict')

    # Buat dictionary dari input pengguna
    data_inf = {
            'admission_count': admission_count,
            'readmission_count': readmission_count,
            'comorbid_conditions_count': comorbid_conditions_count,
            'daily_medication_dosage': daily_medication_dosage,
            'emergency_visit_count': emergency_visit_count,
            'admission_day_of_week': admission_day_of_week, 
            'admission_month': admission_month, 
            'admission_year': admission_year, # Tidak digunakan dalam model
            'hospital_name': hospital_name, # Tidak digunakan dalam model
            'condition_type': condition_type,
            'patient_age_group': patient_age_group,
            'patient_gender': patient_gender,
            'severity_level': severity_level,
            'seasonal_indicator': seasonal_indicator, # Tidak digunakan dalam model
            'primary_diagnosis_code': primary_diagnosis_code

        }
    # Convert dictionary to DataFrame
    data_inf = pd.DataFrame([data_inf])
    st.subheader("Ringkasan Data Input Pasien:")
    data_inf

    if submitted:

        prediction = model_pipeline.predict(data_inf)
        st.subheader('Hasil Prediksi:')
        st.write(f'Perkiraan Rata-Rata Lama Tinggal Pasien: **{prediction[0]:.2f} hari**')

        st.markdown("---")
    st.markdown("""
    \n**Disclaimer:** Prediksi ini berdasarkan model machine learning dan hanya bersifat estimasi.
    Keputusan medis harus selalu dibuat oleh profesional kesehatan.
    """)
    st.caption("Aplikasi Prediksi LoS Pasien | Dibuat oleh: Pradita Ajeng Wiguna (RMT-042) | Proyek Milestone 2 - Hacktiv8")
# Jalankan Aplikasi 
if __name__ == '__main__':
    run()