import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Definisi Data dari Context File ---

# Kolom Fitur (dari column_names.txt)
FEATURE_COLUMNS = ['country', 'market_segment', 'previous_cancellations', 'booking_changes', 'deposit_type', 'days_in_waiting_list', 'customer_type', 'reserved_room_type', 'required_car_parking_spaces', 'total_of_special_requests', 'is_repeat_canceler', 'is_high_risk', 'country_group', 'commitment_score', 'booking_stability']

# Nilai Unik Kategori (dari kolom_kategori_unique_values.csv)
CATEGORY_UNIQUE_VALUES = {
    'market_segment': ['Offline TA/TO', 'Online TA', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Other'],
    'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
    'customer_type': ['Transient-Party', 'Transient', 'Contract', 'Group'],
    'country_group': ['Top_International', 'PRT', 'Other'],
    'reserved_room_type': ['A', 'E', 'D', 'F', 'B', 'G', 'C', 'H', 'L', 'P']
}

# Statistik Numerik (dari kolom_numerik_range.csv)
NUMERIC_RANGES = {
    'commitment_score': {'min': 0.0, 'max': 21.0, 'mean': 0.8571069603819416},
    'booking_stability': {'min': -39.1, 'max': 21.0, 'mean': -0.012159429480813196},
    'previous_cancellations': {'min': 0.0, 'max': 26.0, 'mean': 0.08679836789393704},
    'booking_changes': {'min': 0.0, 'max': 21.0, 'mean': 0.22089670108767184},
    'days_in_waiting_list': {'min': 0.0, 'max': 391.0, 'mean': 2.33056130568485},
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
    # Fitur 'country' di-input secara default di bagian logika prediksi.
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
        st.subheader(f"3. {list(INPUT_GROUPS.keys())