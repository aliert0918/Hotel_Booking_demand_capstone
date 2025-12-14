import streamlit as st
import pandas as pd
import joblib
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Hotel Booking Cancellation Predictor",
    page_icon="hk",
    layout="wide"
)

# --- 1. Load the Model ---
@st.cache_resource
def load_model():
    model_path = 'threshold_tuned_HotelCapstone_FOR_DEPLOYMENT_20251214_22_53.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Please ensure the .pkl file is in the same directory.")
        return None
    return joblib.load(model_path)

model = load_model()

# --- 2. Title and Description ---
st.title("🏨 Hotel Booking Cancellation Prediction")
st.markdown("""
This app predicts the likelihood of a hotel booking cancellation based on customer and booking details.
Please adjust the parameters below and click **Predict**.
""")

st.markdown("---")

# --- 3. Input Form & Feature Grouping ---
# We use a form to prevent reloading on every input change
with st.form("prediction_form"):
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)

    # --- Group A: Customer Profile ---
    with col1:
        st.subheader("👤 Customer Profile")
        
        # - market_segment unique values
        market_segment_opts = ['Offline TA/TO', 'Online TA', 'Direct', 'Groups', 'Corporate', 'Complementary', 'Aviation', 'Other']
        market_segment = st.selectbox("Market Segment", options=market_segment_opts)
        
        # - customer_type unique values
        customer_type_opts = ['Transient-Party', 'Transient', 'Contract', 'Group']
        customer_type = st.selectbox("Customer Type", options=customer_type_opts)
        
        # - country_group unique values (Mapped to 'country' feature)
        country_opts = ['Top_International', 'PRT', 'Other']
        country = st.selectbox("Country / Region", options=country_opts)

    # --- Group B: Booking Details ---
    with col2:
        st.subheader("📅 Booking Details")
        
        # - reserved_room_type unique values
        room_type_opts = ['A', 'E', 'D', 'F', 'B', 'G', 'C', 'H', 'L', 'P']
        reserved_room_type = st.selectbox("Reserved Room Type", options=room_type_opts)
        
        # - deposit_type unique values
        deposit_type_opts = ['No Deposit', 'Non Refund', 'Refundable']
        deposit_type = st.selectbox("Deposit Type", options=deposit_type_opts)
        
        # - days_in_waiting_list range (min: 0, max: 391)
        days_in_waiting_list = st.number_input(
            "Days in Waiting List", 
            min_value=0, 
            max_value=400, 
            value=0,
            step=1
        )

    # --- Group C: History & Requests ---
    with col3:
        st.subheader("📜 History & Requests")
        
        # - previous_cancellations range (min: 0, max: 26)
        previous_cancellations = st.number_input(
            "Previous Cancellations", 
            min_value=0, 
            max_value=30, 
            value=0, 
            step=1
        )
        
        # - booking_changes range (min: 0, max: 21)
        booking_changes = st.number_input(
            "Booking Changes", 
            min_value=0, 
            max_value=25, 
            value=0, 
            step=1
        )
        
        # 'total_of_special_requests' - Not in numeric range file, assuming standard count
        total_of_special_requests = st.number_input(
            "Total Special Requests", 
            min_value=0, 
            max_value=10, 
            value=0, 
            step=1
        )
        
        # 'required_car_parking_spaces' - Not in numeric range file, assuming standard count
        required_car_parking_spaces = st.number_input(
            "Car Parking Spaces Required", 
            min_value=0, 
            max_value=10, 
            value=0, 
            step=1
        )

    st.markdown("---")
    
    # Submit Button
    submit_btn = st.form_submit_button("🚀 Predict Cancellation")

# --- 4. Prediction Logic ---
if submit_btn:
    if model:
        # Construct DataFrame with exact column names from 
        input_data = {
            'country': [country],
            'market_segment': [market_segment],
            'previous_cancellations': [previous_cancellations],
            'booking_changes': [booking_changes],
            'deposit_type': [deposit_type],
            'days_in_waiting_list': [days_in_waiting_list],
            'customer_type': [customer_type],
            'reserved_room_type': [reserved_room_type],
            'required_car_parking_spaces': [required_car_parking_spaces],
            'total_of_special_requests': [total_of_special_requests]
        }
        
        df_input = pd.DataFrame(input_data)

        # Display input data (optional debugging)
        with st.expander("See Input Data"):
            st.dataframe(df_input)

        try:
            # Predict
            prediction = model.predict(df_input)[0]
            
            # Since the model is a TunedThresholdClassifierCV, it might have predict_proba
            try:
                probability = model.predict_proba(df_input)[0][1]
            except:
                probability = None

            # --- 5. Display Results ---
            st.subheader("Prediction Result")
            
            if prediction == 1:
                st.error(f"⚠️ Prediction: **Likely to Cancel**")
                if probability is not None:
                    st.write(f"Confidence: **{probability:.2%}**")
            else:
                st.success(f"✅ Prediction: **Likely to Stay**")
                if probability is not None:
                    st.write(f"Confidence: **{(1-probability):.2%}**")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model not loaded. Please check the file path.")

# --- Footer ---
st.markdown("---")
st.caption("Model: threshold_tuned_HotelCapstone | Deployment Date: Dec 2025")