import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import ast

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide")

# --- 2. ROBUST DATA LOADING ---
@st.cache_resource
def load_resources():
    # 1. Load Model
    model_path = 'threshold_tuned_HotelCapstone_FOR_DEPLOYMENT_20251214_22_53.pkl'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}. Please make sure it is in the same folder.")
        return None, None, None, None

    # 2. Load Column Names (Robust Parsing)
    # File content example: ['country', 'market_segment', ...]
    try:
        with open('column_names.txt', 'r') as f:
            content = f.read().strip()
            # Remove brackets, single quotes, double quotes
            clean_content = content.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
            # Split by comma and strip whitespace from each name
            feature_names = [x.strip() for x in clean_content.split(',') if x.strip()]
    except Exception as e:
        st.error(f"❌ Error reading column_names.txt: {e}")
        return None, None, None, None

    # 3. Load Categorical Values
    cat_options = {}
    try:
        cat_df = pd.read_csv('kolom_kategori_unique_values.csv')
        for _, row in cat_df.iterrows():
            col_name = row['column']
            val_str = str(row['unique_values'])
            
            # Try parsing list string, fallback to simple string split if needed
            try:
                unique_vals = ast.literal_eval(val_str)
            except:
                unique_vals = [x.strip() for x in val_str.replace('[','').replace(']','').replace("'", "").split(',')]
            
            cat_options[col_name] = unique_vals
            
        # Manually ensure binary columns exist if not in CSV
        if 'is_repeat_canceler' not in cat_options: cat_options['is_repeat_canceler'] = [0, 1]
        if 'is_high_risk' not in cat_options: cat_options['is_high_risk'] = [0, 1]

    except Exception as e:
        st.error(f"❌ Error reading kolom_kategori_unique_values.csv: {e}")
        return None, None, None, None

    # 4. Load Numerical Ranges
    num_ranges = {}
    try:
        # Load with index_col=0 so 'min'/'max' become row labels
        num_df = pd.read_csv('kolom_numerik_range.csv', index_col=0)
        
        for col in num_df.columns:
            # Check if min/max exist in the index
            if 'min' in num_df.index and 'max' in num_df.index:
                min_val = num_df.loc['min', col]
                max_val = num_df.loc['max', col]
                num_ranges[col] = (float(min_val), float(max_val))
            else:
                # Fallback if rows are named differently
                num_ranges[col] = (float(num_df[col].min()), float(num_df[col].max()))

        # Defaults for columns that might be missing in range file but present in names
        defaults = {
            'days_in_waiting_list': (0.0, 365.0),
            'required_car_parking_spaces': (0.0, 10.0),
            'total_of_special_requests': (0.0, 10.0),
            'previous_cancellations': (0.0, 30.0),
            'booking_changes': (0.0, 20.0),
            'commitment_score': (0.0, 100.0),
            'booking_stability': (-10.0, 10.0)
        }
        for col, rng in defaults.items():
            if col not in num_ranges:
                num_ranges[col] = rng

    except Exception as e:
        st.error(f"❌ Error reading kolom_numerik_range.csv: {e}")
        return None, None, None, None

    return model, feature_names, cat_options, num_ranges

# Run Loader
model, feature_names, cat_options, num_ranges = load_resources()

# --- 3. LIME SETUP (Synthetic) ---
@st.cache_resource
def get_lime_explainer(_model, _feature_names, _cat_options, _num_ranges):
    if not _feature_names: return None, None

    # Generate synthetic training data for LIME initialization
    # (Since we don't have the original X_train)
    rows = 500
    synthetic_data = {}

    for col in _feature_names:
        if col in _cat_options:
            synthetic_data[col] = np.random.choice(_cat_options[col], rows)
        elif col in _num_ranges:
            mn, mx = _num_ranges[col]
            synthetic_data[col] = np.random.uniform(mn, mx, rows)
        elif col == 'country':
            synthetic_data[col] = np.random.choice(['PRT', 'GBR', 'FRA', 'ESP', 'DEU'], rows)
        else:
            synthetic_data[col] = [0]*rows

    train_df = pd.DataFrame(synthetic_data)
    
    # Encode categoricals for LIME (LIME needs numpy array of numbers)
    categorical_features_indices = []
    categorical_names = {} # Map index -> list of string values
    
    train_encoded = train_df.copy()

    for i, col in enumerate(_feature_names):
        # Identify if column is categorical (in cat options or string type)
        is_cat = (col in _cat_options) or (train_df[col].dtype == 'object')
        
        if is_cat:
            categorical_features_indices.append(i)
            # Get unique values
            if col in _cat_options:
                unique_vals = _cat_options[col]
            else:
                unique_vals = list(train_df[col].unique())
            
            categorical_names[i] = unique_vals
            
            # Map strings to int index
            val_to_idx = {v: k for k, v in enumerate(unique_vals)}
            train_encoded[col] = train_df[col].map(lambda x: val_to_idx.get(x, 0))

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=train_encoded.to_numpy(),
        feature_names=_feature_names,
        class_names=['Not Cancel', 'Cancel'],
        categorical_features=categorical_features_indices,
        categorical_names=categorical_names,
        mode='classification',
        verbose=False
    )
    return explainer, categorical_names

if model:
    explainer, lime_cat_names = get_lime_explainer(model, feature_names, cat_options, num_ranges)

# --- 4. STREAMLIT UI ---
st.title("🏨 Hotel Booking Cancellation")
st.markdown("Predict if a customer will cancel based on booking details.")

if not model:
    st.warning("Please upload the required files to the app directory.")
    st.stop()

# --- INPUT FORM (Grouped) ---
user_inputs = {}

# Define Tabs
tab_booking, tab_customer, tab_history = st.tabs(["📝 Booking Details", "👤 Customer Profile", "📊 History & Scores"])

with tab_booking:
    st.header("Booking Details")
    col1, col2 = st.columns(2)
    
    with col1:
        # Market Segment
        opts = cat_options.get('market_segment', ['Direct', 'Online TA'])
        user_inputs['market_segment'] = st.selectbox("Market Segment", opts)
        
        # Reserved Room Type
        opts = cat_options.get('reserved_room_type', ['A', 'B', 'C'])
        user_inputs['reserved_room_type'] = st.selectbox("Reserved Room Type", opts)
        
        # Deposit Type
        opts = cat_options.get('deposit_type', ['No Deposit', 'Non Refund'])
        user_inputs['deposit_type'] = st.selectbox("Deposit Type", opts)

    with col2:
        # Car Parking
        mn, mx = num_ranges.get('required_car_parking_spaces', (0, 10))
        user_inputs['required_car_parking_spaces'] = st.number_input("Parking Spaces", int(mn), int(mx), 0)
        
        # Special Requests
        mn, mx = num_ranges.get('total_of_special_requests', (0, 10))
        user_inputs['total_of_special_requests'] = st.number_input("Special Requests", int(mn), int(mx), 0)
        
        # Waiting List
        mn, mx = num_ranges.get('days_in_waiting_list', (0, 365))
        user_inputs['days_in_waiting_list'] = st.number_input("Days in Waiting List", int(mn), int(mx), 0)

with tab_customer:
    st.header("Customer Profile")
    col1, col2 = st.columns(2)
    
    with col1:
        # Customer Type
        opts = cat_options.get('customer_type', ['Transient', 'Group'])
        user_inputs['customer_type'] = st.selectbox("Customer Type", opts)
        
        # Country Group
        opts = cat_options.get('country_group', ['PRT', 'Other'])
        user_inputs['country_group'] = st.selectbox("Country Group", opts)
    
    with col2:
        # Country (Free text or select if available)
        user_inputs['country'] = st.text_input("Country Code (e.g. PRT, FRA)", "PRT")

with tab_history:
    st.header("History & Scores")
    col1, col2 = st.columns(2)
    
    with col1:
        # Previous Cancellations
        mn, mx = num_ranges.get('previous_cancellations', (0, 30))
        user_inputs['previous_cancellations'] = st.number_input("Previous Cancellations", int(mn), int(mx), 0)
        
        # Booking Changes
        mn, mx = num_ranges.get('booking_changes', (0, 20))
        user_inputs['booking_changes'] = st.number_input("Booking Changes", int(mn), int(mx), 0)
        
        # Is Repeat Canceler (Binary)
        user_inputs['is_repeat_canceler'] = st.selectbox("Is Repeat Canceler?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        
        # Is High Risk
        user_inputs['is_high_risk'] = st.selectbox("Is High Risk Customer?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with col2:
        # Commitment Score
        mn, mx = num_ranges.get('commitment_score', (0.0, 10.0))
        user_inputs['commitment_score'] = st.slider("Commitment Score", float(mn), float(mx), float(mn))
        
        # Booking Stability
        mn, mx = num_ranges.get('booking_stability', (-5.0, 5.0))
        user_inputs['booking_stability'] = st.slider("Booking Stability", float(mn), float(mx), float(mn))

# --- 5. PREDICTION LOGIC ---
st.markdown("---")
if st.button("🚀 Predict Cancellation", type="primary", use_container_width=True):
    
    # 1. Prepare Input DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Ensure correct column order
    try:
        input_df = input_df[feature_names]
        
        # 2. Predict
        # TunedThresholdClassifierCV usually exposes predict/predict_proba
        pred_class = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        
        # 3. Display
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred_class == 1:
                st.error("### ⚠️ Prediction: Cancel")
                st.write("This booking is likely to be cancelled.")
            else:
                st.success("### ✅ Prediction: Not Cancel")
                st.write("This booking is likely to be fulfilled.")
        
        with col_res2:
            st.metric("Probability of Cancellation", f"{pred_proba[1]:.2%}")
            st.progress(float(pred_proba[1]))

        # 4. LIME Visualization
        st.markdown("---")
        st.subheader("🔍 Why this result?")
        
        with st.spinner("Generating explanation..."):
            
            # --- CUSTOM PREDICT FUNCTION FOR LIME ---
            # LIME gives us a Numpy Array of numbers (ints for categories).
            # We must convert this back to a DataFrame with Strings for the pipeline.
            def predict_fn_lime(np_array):
                # 1. Convert to DataFrame
                temp_df = pd.DataFrame(np_array, columns=feature_names)
                
                # 2. Decode Categoricals (Int -> String)
                for i, col in enumerate(feature_names):
                    if i in lime_cat_names:
                        # Get the list of string options
                        mapping = lime_cat_names[i]
                        # Apply mapping (index -> string)
                        # We use apply to handle the whole column safely
                        temp_df[col] = temp_df[col].apply(
                            lambda x: mapping[int(x)] if 0 <= int(x) < len(mapping) else mapping[0]
                        )
                
                # 3. Pass fully restored DataFrame to Model
                return model.predict_proba(temp_df)

            # --- PREPARE INSTANCE FOR LIME ---
            # We need to turn our single 'input_df' row into the encoded format LIME expects
            lime_instance = []
            for i, col in enumerate(feature_names):
                val = user_inputs[col]
                if i in lime_cat_names:
                    # It's categorical, find the index of the string value
                    try:
                        idx = lime_cat_names[i].index(val)
                    except ValueError:
                        idx = 0 # Default if unknown value
                    lime_instance.append(idx)
                else:
                    # It's numerical
                    lime_instance.append(val)
            
            lime_instance = np.array(lime_instance)

            # --- EXPLAIN ---
            exp = explainer.explain_instance(
                data_row=lime_instance,
                predict_fn=predict_fn_lime,
                num_features=10
            )
            
            # Plot
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            st.caption("Green bars support the prediction, Red bars oppose it.")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write("Columns expected:", feature_names)