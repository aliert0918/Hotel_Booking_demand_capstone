import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Definisi Data dari File yang Diunggah ---
# [cite_start]Kolom Fitur (dari column_names.txt) [cite: 1]
# Dikelompokkan untuk tampilan yang lebih baik di Streamlit
FEATURE_COLUMNS = ['country', 'market_segment', 'previous_cancellations', 'booking_changes', 'deposit_type', 'days_in_waiting_list', 'customer_type', 'reserved_room_type', 'required_car_parking_spaces', 'total_of_special_requests', 'is_repeat_canceler', 'is_high_risk', 'country_group', 'commitment_score', 'booking_stability']

# [cite_start]Nilai Unik Kategori (dari kolom_kategori_unique_values.csv) [cite: 2]
CATEGORY_UNIQUE_VALUES = {
    'market_segment': ['Offline TA/TO', 'Online TA', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Other'],
    'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
    'customer_type': ['Transient-Party', 'Transient', 'Contract', 'Group'],
    'country_group': ['Top_International', 'PRT', 'Other'],
    'reserved_room_type': ['A', 'E', 'D', 'F', 'B', 'G', 'C', 'H', 'L', 'P']
}

# [cite_start]Statistik Numerik (dari kolom_numerik_range.csv) [cite: 3]
NUMERIC_RANGES = {
    'commitment_score': {'min': 0.0, 'max': 21.0, 'mean': 0.8571},
    'booking_stability': {'min': -39.1, 'max': 21.0, 'mean': -0.0122},
    'previous_cancellations': {'min': 0.0, 'max': 26.0, 'mean': 0.0868},
    'booking_changes': {'min': 0.0, 'max': 21.0, 'mean': 0.2209},
    'days_in_waiting_list': {'min': 0.0, 'max': 391.0, 'mean': 2.3306},
    'required_car_parking_spaces': {'min': 0, 'max': 8, 'mean': 0.06}, # Diasumsikan max 8, karena tidak ada di file, tapi fitur ini biasanya kecil. Kita pakai 8 saja.
    'total_of_special_requests': {'min': 0, 'max': 5, 'mean': 0.6}, # Diasumsikan max 5, karena tidak ada di file, tapi fitur ini biasanya kecil. Kita pakai 5 saja.
    'is_repeat_canceler': {'min': 0, 'max': 1, 'mean': 0.003}, # Fitur Biner
    'is_high_risk': {'min': 0, 'max': 1, 'mean': 0.003}, # Fitur Biner
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
    # 'country' diabaikan dalam input UI karena memiliki terlalu banyak nilai unik, 
    # namun harus ada di FEATURE_COLUMNS jika model membutuhkannya
}


# --- 2. Load Model ---
# Ganti dengan path model yang sebenarnya.
MODEL_PATH = 'final_model_capstone_hotel_20251214_22_00.pkl'

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
                    # Input Biner (Checkbox/Radio)
                    # Menggunakan checkbox untuk biner 0/1
                    value = column_container.checkbox(
                        f"**{feature.replace('_', ' ').title()}**",
                        value=False
                    )
                    input_data[feature] = 1 if value else 0
                
                else:
                    # Input Numerik (Slider atau Number Input)
                    stats = NUMERIC_RANGES.get(feature, {'min': 0, 'max': 100, 'mean': 0})
                    
                    if feature in ['previous_cancellations', 'booking_changes', 'required_car_parking_spaces', 'total_of_special_requests']:
                        # Numerik Diskrit/Integer, menggunakan Slider
                        value = column_container.slider(
                            f"**{feature.replace('_', ' ').title()}**",
                            min_value=int(stats['min']),
                            max_value=int(stats['max']),
                            value=int(stats['min']),
                            step=1,
                            help=f"Rentang valid: {stats['min']} - {stats['max']}"
                        )
                        input_data[feature] = value
                    else:
                        # Numerik Kontinu/Float, menggunakan Number Input
                        value = column_container.number_input(
                            f"**{feature.replace('_', ' ').title()}**",
                            min_value=stats['min'],
                            max_value=stats['max'],
                            value=stats['mean'],
                            step=(stats['max'] - stats['min']) / 100,
                            format="%.4f",
                            help=f"Rentang valid: {stats['min']} - {stats['max']}"
                        )
                        input_data[feature] = value

    # Distribusikan input ke kolom
    st.subheader(f"1. {list(INPUT_GROUPS.keys())[0]}", divider='grey')
    create_input_widgets(INPUT_GROUPS[list(INPUT_GROUPS.keys())[0]], col_input_1)
    
    st.subheader(f"2. {list(INPUT_GROUPS.keys())[1]}", divider='grey')
    create_input_widgets(INPUT_GROUPS[list(INPUT_GROUPS.keys())[1]], col_input_2)
    
    st.subheader(f"3. {list(INPUT_GROUPS.keys())[2]}", divider='grey')
    create_input_widgets(INPUT_GROUPS[list(INPUT_GROUPS.keys())[2]], col_input_3)
    
    # [cite_start]Tambahkan fitur 'country' dengan nilai default karena model membutuhkannya [cite: 1]
    # Asumsikan 'country' memiliki nilai default 'PRT' atau 'Other' jika tidak di-input.
    # Jika model Anda memerlukan 'country' yang valid, ganti 'Other' dengan nilai yang sesuai.
    input_data['country'] = 'Other' 
    
    # --- 5. Prediksi dan Output ---
    st.header("Hasil Prediksi")
    
    if st.button("Lakukan Prediksi Risiko"):
        
        # 1. Bentuk DataFrame Input
        # [cite_start]Pastikan urutan kolom sesuai dengan yang diharapkan model [cite: 1]
        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)
        
        # 2. Lakukan Prediksi
        try:
            # Prediksi Probabilitas
            prob_risk = model.predict_proba(input_df)[:, 1][0]
            
            # Prediksi Kelas (menggunakan threshold yang sudah dituning)
            prediction = model.predict(input_df)[0]
            
            st.divider()

            # 3. Tampilkan Hasil
            st.metric(
                label="Probabilitas Risiko Pembatalan", 
                value=f"{prob_risk * 100:.2f} %"
            )
            
            if prediction == 1:
                st.error("🚨 **PREDIKSI: RISIKO TINGGI PEMBATALAN**", icon="🚫")
                st.markdown(
                    "Pemesanan ini memiliki probabilitas **tinggi** untuk dibatalkan berdasarkan ambang batas (*threshold*) model yang telah dioptimalkan."
                )
            else:
                st.success("✅ **PREDIKSI: RISIKO RENDAH PEMBATALAN**", icon="👍")
                st.markdown(
                    "Pemesanan ini memiliki probabilitas **rendah** untuk dibatalkan berdasarkan ambang batas (*threshold*) model yang telah dioptimalkan."
                )
                
            st.markdown("---")
            st.subheader("Data Input yang Digunakan:")
            st.dataframe(input_df.style.format(precision=4), use_container_width=True)
            
        except Exception as e:
            st.error(f"Terjadi error saat melakukan prediksi: {e}")
            st.warning("Pastikan input data Anda valid dan model telah dimuat dengan benar.")

if __name__ == "__main__":
    main()