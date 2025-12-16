import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# --- 0. HARDCODED DATA (Menggantikan file .txt dan .csv) ---

FEATURE_NAMES = [
    'country', 'market_segment', 'previous_cancellations', 'booking_changes', 
    'deposit_type', 'days_in_waiting_list', 'customer_type', 'reserved_room_type', 
    'required_car_parking_spaces', 'total_of_special_requests', 'is_repeat_canceler', 
    'is_high_risk', 'country_group', 'commitment_score', 'booking_stability'
]

CAT_OPTIONS = {
    'market_segment': ['Offline TA/TO', 'Online TA', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Other'],
    'deposit_type': ['No Deposit', 'Non Refund', 'Refundable'],
    'customer_type': ['Transient-Party', 'Transient', 'Contract', 'Group'],
    'country_group': ['Top_International', 'PRT', 'Other'],
    'reserved_room_type': ['A', 'E', 'D', 'F', 'B', 'G', 'C', 'H', 'L', 'P'],
    'is_repeat_canceler': [0, 1],
    'is_high_risk': [0, 1]
}

NUM_RANGES = {
    'commitment_score': (0.0, 21.0),
    'booking_stability': (-39.1, 21.0),
    'previous_cancellations': (0.0, 26.0),
    'booking_changes': (0.0, 21.0),
    'days_in_waiting_list': (0.0, 391.0),
    'required_car_parking_spaces': (0.0, 10.0),
    'total_of_special_requests': (0.0, 10.0),
}


# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide")


# --- 2. MODEL LOADING ---
@st.cache_resource
def load_resources(model_path):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}. Pastikan file model .pkl ada di folder yang sama.")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

    return model, FEATURE_NAMES

model_file_path = 'threshold_tuned_HotelCapstone_FOR_DEPLOYMENT_20251214_22_53.pkl'
model, feature_names = load_resources(model_file_path)


# --- 3. LIME SETUP (Synthetic) ---
@st.cache_resource
def get_lime_explainer(_model, _feature_names):
    if not _feature_names: return None, None

    rows = 500
    synthetic_data = {}

    for col in _feature_names:
        if col in CAT_OPTIONS:
            synthetic_data[col] = np.random.choice(CAT_OPTIONS[col], rows)
        elif col in NUM_RANGES:
            mn, mx = NUM_RANGES[col]
            synthetic_data[col] = np.random.uniform(mn, mx, rows)
        else:
            # Country: 'country' is not in CAT_OPTIONS, handle it explicitly as categorical
            synthetic_data[col] = np.random.choice(['PRT', 'GBR', 'FRA'], rows) if col == 'country' else np.zeros(rows)

    train_df = pd.DataFrame(synthetic_data)
    
    # Memastikan Tipe Data Integer untuk Kolom yang Sesuai
    int_cols = ['required_car_parking_spaces', 'total_of_special_requests', 
                'previous_cancellations', 'booking_changes', 'days_in_waiting_list']
    for col in int_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].round().astype(int) 

    # Prepare data for LIME
    categorical_features_indices = []
    categorical_names = {}
    train_encoded = train_df.copy()

    for i, col in enumerate(_feature_names):
        # Identifikasi kolom yang kategorial (ada di CAT_OPTIONS atau bertipe 'object', seperti 'country')
        is_cat = (col in CAT_OPTIONS) or (train_df[col].dtype == 'object')
        
        if is_cat:
            categorical_features_indices.append(i)
            
            # Gunakan options yang sudah di-hardcode, atau ambil dari data sintetis (untuk 'country')
            unique_vals = CAT_OPTIONS.get(col, list(train_df[col].unique()))
            
            categorical_names[i] = unique_vals
            
            val_to_idx = {v: k for k, v in enumerate(unique_vals)}
            train_encoded[col] = train_df[col].map(lambda x: val_to_idx.get(x, 0))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_encoded.to_numpy(),
        feature_names=_feature_names,
        class_names=['Not Cancel', 'Cancel'],
        categorical_features=categorical_features_indices,
        categorical_names=categorical_names,
        mode='classification',
        verbose=False,
    )
    return explainer, categorical_names

if model:
    explainer, lime_cat_names = get_lime_explainer(model, feature_names)

# --- 4. STREAMLIT UI (Kode UI sama dengan sebelumnya) ---
st.title("🏨 Hotel Booking Cancellation Predictor")
st.markdown("Masukkan detail pemesanan di bawah ini:")

if not model:
    st.stop()

# --- INPUT FORM (Grouped) ---
user_inputs = {}
tab_booking, tab_customer, tab_history = st.tabs(["📝 Booking Details", "👤 Customer Profile", "📊 History & Scores"])

with tab_booking:
    st.header("Booking Details")
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['market_segment'] = st.selectbox("Market Segment", CAT_OPTIONS['market_segment'])
        user_inputs['reserved_room_type'] = st.selectbox("Reserved Room Type", CAT_OPTIONS['reserved_room_type'])
        user_inputs['deposit_type'] = st.selectbox("Deposit Type", CAT_OPTIONS['deposit_type'])
    with col2:
        mn, mx = NUM_RANGES.get('required_car_parking_spaces', (0, 10))
        user_inputs['required_car_parking_spaces'] = st.number_input("Parking Spaces", int(mn), int(mx), 0)
        mn, mx = NUM_RANGES.get('total_of_special_requests', (0, 10))
        user_inputs['total_of_special_requests'] = st.number_input("Special Requests", int(mn), int(mx), 0)
        mn, mx = NUM_RANGES.get('days_in_waiting_list', (0, 391))
        user_inputs['days_in_waiting_list'] = st.number_input("Days in Waiting List", int(mn), int(mx), 0)

with tab_customer:
    st.header("Customer Profile")
    col1, col2 = st.columns(2)
    with col1:
        user_inputs['customer_type'] = st.selectbox("Customer Type", CAT_OPTIONS['customer_type'])
        user_inputs['country_group'] = st.selectbox("Country Group", CAT_OPTIONS['country_group'])
    with col2:
        user_inputs['country'] = st.text_input("Country Code (e.g. PRT, FRA)", "PRT")

with tab_history:
    st.header("History & Scores")
    col1, col2 = st.columns(2)
    with col1:
        mn, mx = NUM_RANGES.get('previous_cancellations', (0, 26))
        user_inputs['previous_cancellations'] = st.number_input("Previous Cancellations", int(mn), int(mx), 0)
        mn, mx = NUM_RANGES.get('booking_changes', (0, 21))
        user_inputs['booking_changes'] = st.number_input("Booking Changes", int(mn), int(mx), 0)
        user_inputs['is_repeat_canceler'] = st.selectbox("Is Repeat Canceler?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        user_inputs['is_high_risk'] = st.selectbox("Is High Risk Customer?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col2:
        mn, mx = NUM_RANGES.get('commitment_score', (0.0, 21.0))
        user_inputs['commitment_score'] = st.slider("Commitment Score", float(mn), float(mx), float(mn), step=0.1)
        mn, mx = NUM_RANGES.get('booking_stability', (-39.1, 21.0))
        user_inputs['booking_stability'] = st.slider("Booking Stability", float(mn), float(mx), 0.0, step=0.1)


# --- 5. PREDICTION LOGIC ---
st.markdown("---")
if st.button("🚀 Predict Cancellation", type="primary", use_container_width=True):
    
    # 1. Prepare Input DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    try:
        input_df = input_df[feature_names]
        pred_class = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        
        # 3. Display Results
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred_class == 1:
                st.error("### ⚠️ Prediction: Cancelled")
            else:
                st.success("### ✅ Prediction: Not Cancelled")
        
        with col_res2:
            st.metric("Probability of Cancellation", f"{pred_proba[1]:.2%}")
            st.progress(float(pred_proba[1]))

        # 4. LIME Visualization
        st.markdown("---")
        st.subheader("🔍 LIME Explanation (Why this result?)")
        
        with st.spinner("Generating explanation..."):
            
            # --- CUSTOM PREDICT FUNCTION FOR LIME (Perbaikan Utama di sini) ---
            def predict_fn_lime(np_array):
                temp_df = pd.DataFrame(np_array, columns=feature_names)
                
                for i, col in enumerate(feature_names):
                    # PERBAIKAN: Cek index kolom (i) di lime_cat_names untuk semua kolom kategorial
                    # Ini akan mencakup 'country' dan semua kolom di CAT_OPTIONS
                    if i in lime_cat_names:
                        mapping = lime_cat_names[i]
                        temp_df[col] = temp_df[col].apply(
                            lambda x: mapping[int(round(x))] if 0 <= int(round(x)) < len(mapping) else mapping[0]
                        )
                    # Jika bukan kategori (numerik), biarkan nilainya (yang sudah berupa float/angka)
                
                return model.predict_proba(temp_df)

            # PREPARE INSTANCE FOR LIME (String -> Numeric Index encoding)
            lime_instance = []
            for i, col in enumerate(feature_names):
                val = user_inputs[col]
                
                # Cek index kolom (i) di lime_cat_names untuk encoding
                if i in lime_cat_names: 
                    mapping = lime_cat_names[i]
                    # LIME instance perlu angka indeks
                    try:
                        idx = mapping.index(val)
                    except ValueError:
                        idx = 0 
                    lime_instance.append(idx)
                else:
                    # Numerik
                    lime_instance.append(val)
            
            lime_instance = np.array(lime_instance)

            # EXPLAIN
            exp = explainer.explain_instance(
                data_row=lime_instance,
                predict_fn=predict_fn_lime,
                num_features=10
            )
            
            # Plot
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            st.caption("Green bars mendukung prediksi (Cancellation/Not Cancelled), Red bars menentangnya.")

    except Exception as e:
        st.error(f"An unexpected error occurred during prediction/LIME generation: {e}")
        st.warning("Jika error berulang, pastikan semua nilai input sesuai dengan range atau opsi yang tersedia.")
