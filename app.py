import streamlit as st
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


try:
    model = joblib.load('habit_vs_exam_model.pkl')
except FileNotFoundError:
    st.error("File model 'habit_vs_exam_model.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

st.title("Prediksi Kelulusan Ujian Berdasarkan Kebiasaan")
st.write("Aplikasi ini memprediksi apakah seorang mahasiswa akan lulus (nilai >= 70) berdasarkan 8 fitur kebiasaan belajar dan gaya hidup.")


data_dummy = {
    'age': [20.5], 'gender': [0], 'study_hours_per_day': [3.55],
    'social_media_hours': [2.51], 'netflix_hours': [1.82], 'part_time_job': [0],
    'attendance_percentage': [84.13], 'sleep_hours': [6.47], 'diet_quality': [1],
    'exercise_frequency': [3.04], 'parental_education_level': [1], 'internet_quality': [1],
    'mental_health_rating': [5.44], 'extracurricular_participation': [0]
}
df_dummy = pd.DataFrame(data_dummy)


scaler = StandardScaler().fit(df_dummy)


pel_encoder = LabelEncoder().fit(['Bachelor', 'High School', 'Master'])



with st.form("Form_prediksi_kelulusan"):
    st.header("Masukkan data kebiasaan mahasiswa:")
    
   
    col1, col2 = st.columns(2)

    with col1:
        study_hours_per_day = st.number_input('Jam Belajar per Hari', min_value=0.0, max_value=10.0, value=3.5, step=0.1, format="%.1f")
        social_media_hours = st.number_input('Jam Media Sosial per Hari', min_value=0.0, max_value=8.0, value=2.5, step=0.1, format="%.1f")
        netflix_hours = st.number_input('Jam Nonton Netflix per Hari', min_value=0.0, max_value=8.0, value=1.8, step=0.1, format="%.1f")
        attendance_percentage = st.slider('Persentase Kehadiran (%)', min_value=0, max_value=100, value=85)

    with col2:
        sleep_hours = st.number_input('Jam Tidur per Hari', min_value=3.0, max_value=10.0, value=6.5, step=0.1, format="%.1f")
        exercise_frequency = st.slider('Frekuensi Olahraga (kali per minggu)', min_value=0, max_value=6, value=3)
        parental_education_level = st.selectbox('Pendidikan Terakhir Orang Tua', options=['High School', 'Bachelor', 'Master'])
        mental_health_rating = st.slider('Peringkat Kesehatan Mental (1-10)', min_value=1, max_value=10, value=5)

    
    submit = st.form_submit_button("Prediksi Kelulusan")


if submit:
    
    parental_education_encoded = pel_encoder.transform([parental_education_level])[0]

    
    user_input_df = pd.DataFrame({
        'age': [20.5], # Menggunakan nilai rata-rata dari notebook
        'gender': [0], # Dummy value
        'study_hours_per_day': [study_hours_per_day],
        'social_media_hours': [social_media_hours],
        'netflix_hours': [netflix_hours],
        'part_time_job': [0], # Dummy value
        'attendance_percentage': [attendance_percentage],
        'sleep_hours': [sleep_hours],
        'diet_quality': [1], # Dummy value
        'exercise_frequency': [exercise_frequency],
        'parental_education_level': [parental_education_encoded],
        'internet_quality': [1], # Dummy value
        'mental_health_rating': [mental_health_rating],
        'extracurricular_participation': [0] # Dummy value
    })

   
    scaled_features = scaler.transform(user_input_df)
    

    scaled_df = pd.DataFrame(scaled_features, columns=user_input_df.columns)

   
    final_features = scaled_df[[
        'study_hours_per_day', 'social_media_hours', 'netflix_hours',
        'attendance_percentage', 'sleep_hours', 'exercise_frequency',
        'parental_education_level', 'mental_health_rating'
    ]]

    
    prediction = model.predict(final_features)[0]
    prediction_proba = model.predict_proba(final_features)

    st.header("Hasil Prediksi")
    if prediction == 1:
        st.success("Hasil: **LULUS** (Nilai >= 70)")
        st.write(f"Probabilitas kelulusan: **{prediction_proba[0][1]*100:.2f}%**")
    else:
        st.error("Hasil: **TIDAK LULUS** (Nilai < 70)")
        st.write(f"Probabilitas kelulusan: **{prediction_proba[0][1]*100:.2f}%**")

    with st.expander("Lihat Detail Data yang Diproses"):
        st.write("Data yang Anda masukkan:")
        st.write(user_input_df[[
            'study_hours_per_day', 'social_media_hours', 'netflix_hours',
            'attendance_percentage', 'sleep_hours', 'exercise_frequency',
            'parental_education_level', 'mental_health_rating'
        ]])
        st.write("Data setelah di-scale (input untuk model):")
        st.write(final_features)
