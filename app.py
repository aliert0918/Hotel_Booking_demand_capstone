import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Definisi Data dari Context File ---

# Kolom Fitur (dari column_names.txt)
[cite_start]FEATURE_COLUMNS = ['country', 'market_segment', 'previous_cancellations', 'booking_changes', 'deposit_type', 'days_in_waiting_list', 'customer_type', 'reserved_room_type', 'required_car_parking_spaces', 'total_of_special_requests', 'is_repeat_canceler', 'is_high_risk', 'country_group', 'commitment_score', 'booking_stability'] # [cite: 1]

# Nilai Unik Kategori (dari kolom_kategori_unique_values.csv)
CATEGORY_UNIQUE_VALUES = {
    # Baris yang dipermasalahkan (Sitasi dipindahkan ke komentar)
    [cite_start]'market_segment': ['Offline TA/TO', 'Online TA', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Other'], # [cite: 2]
    [cite_start]'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'], # [cite: 2]
    [cite_start]'customer_type': ['Transient-Party', 'Transient', 'Contract', 'Group'], # [cite: 2]
    [cite_start]'country_group': ['Top_International', 'PRT', 'Other'], # [cite: 2]
    [cite_start]'reserved_room_type': ['A', 'E', 'D', 'F', 'B', 'G', 'C', 'H', 'L', 'P'] # [cite: 2]
}

# Statistik Numerik (dari kolom_numerik_range.csv)
NUMERIC_RANGES = {
    [cite_start]'commitment_score': {'min': 0.0, 'max': 21.0, 'mean': 0.8571069603819416}, # [cite: 3]
    [cite_start]'booking_stability': {'min': -39.1, 'max': 21.0, 'mean': -0.012159429480813196}, # [cite: 3]
    [cite_start]'previous_cancellations': {'min': 0.0, 'max': 26.0, 'mean': 0.08679836789393704}, # [cite: 3]
    [cite_start]'booking_changes': {'min': 0.0, 'max': 21.0, 'mean': 0.22089670108767184}, # [cite: 3]
    [cite_start]'days_in_waiting_list': {'min': 0.0, 'max': 391.0, 'mean': 2.33056130568485}, # [cite: 3]
    # Nilai asumsi untuk fitur biner/diskret lainnya (sesuai best practice, karena tidak di kolom_numerik_range.csv)
    'required_car_parking_spaces': {'min': 0, 'max': 8, 'mean': 0},
    'total_of_special_requests': {'min': 0, 'max': 5, 'mean': 0},
    'is_repeat_canceler': {'min': 0, 'max': 1, 'mean': 0}, 
    'is_high_risk': {'min': 0, 'max': 1, 'mean': 0}, 
}

# Pengelompokan Fitur untuk Tampilan Streamlit
INPUT_GROUPS = {
    "Status Riwayat Pelanggan": [
        'is_repeat_canceler', 'is_high_risk', 'previous_cancellations', 'commitment_score', 'booking_stability'
    ],
    "Detail Pemesanan": [
        'market_segment', 'country_group', 'deposit_type', 'reserved_room_type', 'days_in_waiting_list', 'booking_changes'
    ],
    "Preferensi Khusus & Tipe Customer": [
        'customer_type', 'required_car_parking_spaces', 'total_of_special_requests'
    ],
}


# --- 2. Load Model ---
# Menggunakan nama model yang diperbarui
MODEL_PATH = 'threshold_tuned_HotelCapstone_FOR_DEPLOYMENT_20251214_22_53.pkl'

@st.cache_resource
def load_model(path):
    """Memuat model menggunakan joblib dengan caching."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Error: File model tidak ditemukan di path: {path}")
        return None
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Muat model saat aplikasi dimulai
model = load_model(MODEL_PATH)

# --- 3. Fungsi Utama Streamlit App ---

def main():
    """Mengatur tampilan dan logika aplikasi Streamlit."""
    
    st.set_page_config(
        page_title="Prediksi Risiko Pembatalan Pemesanan Hotel",
        layout="wide"
    )

    st.title("🏨 Model Prediksi Risiko Pembatalan Pemesanan Hotel")
    st.markdown("Aplikasi ini menggunakan model **TunedThresholdClassifierCV** untuk memprediksi risiko pembatalan pemesanan baru.")
    
    if model is None:
        st.stop()

    # Inisialisasi dictionary untuk menyimpan input pengguna
    input_data = {}
    
    # --- 4. Tampilan Input Fitur Terkelompok ---
    st.header("Masukkan Detail Pemesanan Baru")
    st.markdown("Sesuaikan parameter di bawah ini untuk melihat prediksi risiko pembatalan.")
    
    # Menggunakan layout kolom untuk tampilan yang lebih rapi
    col_input_1, col_input_2, col_input_3 = st.columns(3)
    
    # Fungsi pembantu untuk membuat widget input berdasarkan grup
    def create_input_widgets(column_list, column_container):
        for feature in column_list:
            # Menggunakan kontainer kolom yang diberikan
            with column_container:
                # Logika pembuatan widget
                if feature in CATEGORY_UNIQUE_VALUES:
                    # Input Kategorikal (Selectbox)
                    value = column_container.selectbox(
                        f"**{feature.replace('_', ' ').title()}**",
                        options=CATEGORY_UNIQUE_VALUES[feature]
                    )
                    input_data[feature] = value
                
                elif feature in ['is_repeat_canceler', 'is_high_risk']:
                    # Input Biner (Checkbox)
                    value = column_container.checkbox(
                        f"**{feature.replace('_', ' ').title()}** (1=Ya, 0=Tidak)",
                        value=False
                    )
                    input_data[feature] = 1 if value else 0
                
                else:
                    # Input Numerik (Slider atau Number Input)
                    stats = NUMERIC_RANGES.get(feature, {'min': 0, 'max': 100, 'mean': 0})
                    
                    if feature in ['previous_cancellations', 'booking_changes', 'required_car_parking_spaces', 'total_of_special_requests', 'days_in_waiting_list']:
                        # Numerik Diskrit/Integer, menggunakan Slider
                        # Menggunakan nilai min sebagai default
                        default_value = int(stats['min'])
                        
                        value = column_container.slider(
                            f"**{feature.replace('_', ' ').title()}**",
                            min_value=int(stats['min']),
                            max_value=int(stats['max']),
                            value=default_value,
                            step=1,
                            help=f"Rentang valid: {stats['min']} - {stats['max']}"
                        )
                        input_data[feature] = value
                    else:
                        # Numerik Kontinu/Float, menggunakan Number Input
                        # Menggunakan nilai rata-rata (mean) sebagai default
                        default_value = stats['mean']
                        
                        value = column_container.number_input(
                            f"**{feature.replace('_', ' ').title()}**",
                            min_value=stats['min'],
                            max_value=stats['max'],
                            value=default_value,
                            step=0.01,
                            format="%.4f",
                            help=f"Rentang valid: {stats['min']} - {stats['max']}"
                        )
                        input_data[feature] = value

    # Distribusikan input ke kolom berdasarkan grup
    
    with col_input_1:
        st.subheader(f"1. {list(INPUT_GROUPS.keys())[0]}", divider='grey')
        create_input_widgets(INPUT_GROUPS[list(INPUT_GROUPS.keys())[0]], col_input_1)
    
    with col_input_2:
        st.subheader(f"2. {list(INPUT_GROUPS.keys())[1]}", divider='grey')
        create_input_widgets(INPUT_GROUPS[list(INPUT_GROUPS.keys())[1]], col_input_2)
    
    with col_input_3:
        st.subheader(f"3. {list(INPUT_GROUPS.keys())[2]}", divider='grey')
        create_input_widgets(INPUT_GROUPS[list(INPUT_GROUPS.keys())[2]], col_input_3)
    
    # Tambahkan fitur 'country' dengan nilai default.
    input_data['country'] = 'Other' 
    
    # --- 5. Prediksi dan Output ---
    st.header("Hasil Prediksi")
    
    if st.button("Lakukan Prediksi Risiko", type="primary"):
        
        # 1. Bentuk DataFrame Input
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # 2. Lakukan Prediksi
        with st.spinner("Memproses prediksi..."):
            try:
                # Prediksi Probabilitas
                prob_risk = model.predict_proba(input_df)[:, 1][0]
                
                # Prediksi Kelas (menggunakan threshold yang sudah dituning)
                prediction = model.predict(input_df)[0]
                
                st.divider()

                # 3. Tampilkan Hasil
                
                color_style = "color: red;" if prob_risk > 0.5 else "color: green;"
                
                st.markdown(f"**Probabilitas Risiko Pembatalan:** <span style='{color_style} font-size: 24px;'>{prob_risk * 100:.2f} %</span>", unsafe_allow_html=True)
                
                if prediction == 1:
                    st.error("🚨 **PREDIKSI: RISIKO TINGGI PEMBATALAN**", icon="🚫")
                    st.markdown(
                        "Pemesanan ini diprediksi memiliki **risiko tinggi** untuk dibatalkan berdasarkan ambang batas (*threshold*) model yang telah dioptimalkan."
                    )
                else:
                    st.success("✅ **PREDIKSI: RISIKO RENDAH PEMBATALAN**", icon="👍")
                    st.markdown(
                        "Pemesanan ini diprediksi memiliki **risiko rendah** untuk dibatalkan."
                    )
                    
                st.markdown("---")
                st.subheader("Data Input yang Digunakan:")
                st.dataframe(input_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Terjadi error saat melakukan prediksi: {e}")
                st.warning("Pastikan input data Anda valid dan model telah dimuat dengan benar.")

if __name__ == "__main__":
    main()