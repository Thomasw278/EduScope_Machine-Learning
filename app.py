import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================================
# ====== CONFIGURATION & THEME =============
# ==========================================
st.set_page_config(
    page_title="EduScope : Prediksi  Nilai  Ujian  dan  Motivasi  Siswa",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
    <style>
    .prediction-card {
        background-color: white;
        color: black;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-top: 15px;
    }
    .prediction-card-success {
        background-color: white;
        color: black;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2e7d32;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)




# ==========================================
# ======== LOAD MODEL & SCALER =============
# ==========================================
def load_models():
    model_linear = joblib.load('model_linear_exam.pkl') 
    model_logistic = joblib.load('model_logistic_motivation.pkl')
    scaler = joblib.load('scaler_model.pkl')
    return model_linear, model_logistic, scaler

try:
    model_linear, model_logistic, scaler = load_models()
except FileNotFoundError:
    st.error("⚠️ File model (.pkl) Not Found")

# ==========================================
# ============ UI SIDEBAR ==================
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=80)
    st.title("Kolom Navigasi")
    st.write("Pilih Model Machine Learning dibawah")
    menu = st.radio(
        "Pilih Model Prediksi:", 
        ["Prediksi Nilai Ujian (Regression)", "Prediksi Tingkat Motivasi (Classification)"]
    )
    st.write("---")
    st.caption("Project Machine Learning - Semester 6")





# ==========================================
# ====== MENU : PREDIKSI NILAI UJIAN =======
# ==========================================
if menu == "Prediksi Nilai Ujian (Regression)":
    st.title("🎓 EduScope: Nilai Ujian Siswa")
    st.write("Prediksi nilai *Exam Score* akhir siswa berdasarkan Jam Belajar, Nilai Ujian Sebelumnya, Kehadiran, Keterlibatan Orang Tua, dan Akses Belajar")
    st.write("---")
    st.subheader("Input Parameter Siswa")
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.number_input("Jam Belajar per Minggu (Hours Studied):", min_value=0, max_value=100, value=20)
        previous_score = st.number_input("Nilai Ujian Sebelumnya (Previous Score):", min_value=0, max_value=100, value=75)
        
    with col2:
        parental_opt = st.selectbox("Keterlibatan Orang Tua:", ["Low", "Medium", "High"])
        resources_opt = st.selectbox("Akses Fasilitas Belajar:", ["Low", "Medium", "High"])

        mapping = {"Low": 0, "Medium": 1, "High": 2}
        parental_encoded = mapping[parental_opt]
        resources_encoded = mapping[resources_opt]
        
    attendance = st.slider("Persentase Kehadiran (Attendance %):", min_value=0, max_value=100, value=85)
    
    st.write("")
    if st.button("Hitung Prediksi", use_container_width=True):
        with st.spinner('Sedang menghitung estimasi matriks...'):
            
            input_data = np.array([[hours_studied, attendance, previous_score, parental_encoded, resources_encoded]])
            prediksi = model_linear.predict(input_data)
            final_score = max(0, min(100, prediksi[0]))
            
            st.write("---")
            st.subheader("🎯 Hasil Analisis Model")
            st.metric(label="Estimasi Nilai Ujian Akhir (Exam Score)", value=f"{round(final_score, 2)} Poin")
            st.markdown(
                f"""
                <div class="prediction-card-success">
                    <h4>Kesimpulan Evaluasi:</h4>
                    <p>Siswa dengan intensitas belajar <b>{hours_studied} jam/minggu</b> dan kehadiran kelas <b>{attendance}%</b> 
                    diperkirakan memperoleh capaian nilai akhir sebesar <b>{round(final_score, 2)}</b>.</p>
                    <small><i>*Model ini memiliki rata-rata batas toleransi melesat (MAE) ±1.06 poin.</i></small>
                </div>
                """, 
                unsafe_allow_html=True
            )





# ============================================
# ===== MENU : PREDIKSI TINGKAT MOTIVASI =====
# ============================================
elif menu == "Prediksi Tingkat Motivasi (Classification)":
    st.title("🎓 EduScope: Tingkat Motivasi Siswa")
    st.write("Klasifikasikan tingkat psikologis motivasi siswa berdasarkan data performa lingkungan dan akademis.")
    st.write("---")

    st.subheader("Input Parameter Siswa")
    col1, col2 = st.columns(2)
    
    with col1:
        parental_opt = st.selectbox("Keterlibatan Orang Tua:", ["Low", "Medium", "High"], index=1)
        resources_opt = st.selectbox("Akses Fasilitas Belajar:", ["Low", "Medium", "High"], index=1)
        income_opt = st.selectbox("Pendapatan Keluarga:", ["Low", "Medium", "High"], index=1)
        
    with col2:
        teacher_opt = st.selectbox("Kualitas Guru di Sekolah:", ["Low", "Medium", "High"], index=1)
        peer_opt = st.selectbox("Pengaruh Lingkungan Teman:", ["Negative", "Neutral", "Positive"], index=1)
        previous_score = st.slider("Nilai Ujian Sebelumnya (Previous Score):", min_value=0, max_value=100, value=75)

    mapping_standard = {"Low": 0, "Medium": 1, "High": 2}
    mapping_peer = {"Negative": 0, "Neutral": 1, "Positive": 2}
    
    st.write("")
    if st.button("Prediksi Profil Motivasi", use_container_width=True):
        with st.spinner('Menghitung profil klasifikasi...'):
            
            raw_input = np.array([[
                mapping_standard[parental_opt],
                mapping_standard[resources_opt],
                mapping_standard[income_opt],
                mapping_standard[teacher_opt],
                mapping_peer[peer_opt],
                previous_score
            ]])
            
            input_data_scaled = scaler.transform(raw_input)
            prediksi_kelas = model_logistic.predict(input_data_scaled)

            kategori = {
                0: {"teks": "Low (Rendah)", "emoji": "🔴", "desc": "Siswa membutuhkan perhatian atau intervensi khusus dari pihak bimbingan konseling."}, 
                1: {"teks": "Medium (Sedang)", "emoji": "🟡", "desc": "Tingkat motivasi siswa berada di batas stabil, namun masih bisa dioptimalkan."}, 
                2: {"teks": "High (Tinggi)", "emoji": "🟢", "desc": "Siswa memiliki kemandirian dan determinasi belajar yang sangat tinggi."}
            }
            
            hasil = kategori[prediksi_kelas[0]]
            
            st.write("---")
            st.subheader("🎯 Hasil Klasifikasi Profil")
            
            if prediksi_kelas[0] == 2:
                st.success(f"### {hasil['emoji']} {hasil['teks']}")
            elif prediksi_kelas[0] == 1:
                st.warning(f"### {hasil['emoji']} {hasil['teks']}")
            else:
                st.error(f"### {hasil['emoji']} {hasil['teks']}")
                
            st.markdown(
                f"""
                <div class="prediction-card">
                    <h4>Rekomendasi & Analisis Akademis:</h4>
                    <p>Berdasarkan kecocokan algoritma <b>Logistic Regression</b>, siswa dengan profil parameter yang dimasukkan 
                    menunjukkan indeks orientasi dorongan belajar tingkat <b>{hasil['teks']}</b>.</p>
                    <p style="color: #4a4e69; font-size: 0.95rem;"><b>Catatan Kondisi:</b> {hasil['desc']}</p>
                    <small><i>*Model dijalankan menggunakan 6 fitur prediktor optimal dengan akurasi seimbang kelas sebesar 38%.</i></small>
                </div>
                """, 
                unsafe_allow_html=True
            )