import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import ast

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Hotel Booking Cancellation Predictor", layout="wide")

# --- 2. DATA LOADING & PARSING ---
@st.cache_resource
def load_resources():
    # 1. Load Model
    # Ensure this file is in the same directory
    model_path = 'threshold_tuned_HotelCapstone_FOR_DEPLOYMENT_20251214_22_53.pkl'
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found.")
        return None, None, None, None

    # 2. Load Column Names (Format: ['col1', 'col2', ...])
    try:
        with open('column_names.txt', 'r') as f:
            content = f.read().strip()
            # specific parsing for the list string format provided
            feature_names = ast.literal_eval(content)
    except Exception as e:
        st.error(f"Error reading column_names.txt: {e}")
        return None, None, None, None

    # 3. Load Categorical Values (Format: column, unique_values=['a','b'])
    cat_options = {}
    try:
        cat_df = pd.read_csv('kolom_kategori_unique_values.csv')
        for _, row in cat_df.iterrows():
            col_name = row['column']
            # Parse string representation of list "['a', 'b']" -> ['a', 'b']
            try:
                unique_vals = ast.literal_eval(row['unique_values'])
                cat_options[col_name] = unique_vals
            except:
                # Fallback if eval fails, treat as string split
                cat_options[col_name] = [str(x).strip() for x in str(row['unique_values']).split(',')]
        
        # Manually add binary/categorical columns that might be missing from the CSV but are in column_names
        # Based on your file, 'is_repeat_canceler' and 'is_high_risk' are likely binary 0/1
        if 'is_repeat_canceler' not in cat_options:
            cat_options['is_repeat_canceler'] = [0, 1]
        if 'is_high_risk' not in cat_options:
            cat_options['is_high_risk'] = [0, 1]
        # 'country' is in column_names but not in unique_values.csv. 
        # We will handle it as a text input or generic list in the UI.

    except Exception as e:
        st.error(f"Error reading kolom_kategori_unique_values.csv: {e}")
        return None, None, None, None

    # 4. Load Numerical Ranges (Format: Transposed stats matrix)
    num_ranges = {}
    try:
        # Provide index_col=0 so 'min', 'max' become the index
        num_df = pd.read_csv('kolom_numerik_range.csv', index_col=0)
        
        # Iterate over columns in the dataframe (which are the features)
        for col in num_df.columns:
            min_val = num_df.loc['min', col]
            max_val = num_df.loc['max', col]
            num_ranges[col] = (float(min_val), float(max_val))
            
        # Defaults for numericals that might be missing in the csv but present in feature_names
        # (e.g. required_car_parking_spaces, total_of_special_requests)
        defaults = {
            'required_car_parking_spaces': (0.0, 10.0),
            'total_of_special_requests': (0.0, 10.0),
            'days_in_waiting_list': (0.0, 391.0) # From your file context
        }
        for col, rng in defaults.items():
            if col not in num_ranges:
                num_ranges[col] = rng

    except Exception as e:
        st.error(f"Error reading kolom_numerik_range.csv: {e}")
        return None, None, None, None

    return model, feature_names, cat_options, num_ranges

# Initialize
model, feature_names, cat_options, num_ranges = load_resources()

# --- 3. LIME SETUP ---
@st.cache_resource
def get_lime_explainer(_model, _feature_names, _cat_options, _num_ranges):
    """
    Creates a LIME explainer using synthetic data derived from min/max/unique values.
    """
    rows = 500  # Number of synthetic samples
    synthetic_data = {}

    for col in _feature_names:
        if col in _cat_options:
            # Randomly select from unique values
            synthetic_data[col] = np.random.choice(_cat_options[col], rows)
        elif col in _num_ranges:
            # Randomly sample from uniform distribution
            mn, mx = _num_ranges[col]
            synthetic_data[col] = np.random.uniform(mn, mx, rows)
        elif col == 'country':
            # Fallback for country if missing from options
            synthetic_data[col] = np.random.choice(['PRT', 'GBR', 'FRA', 'ESP', 'DEU'], rows)
        else:
            # Final Fallback
            synthetic_data[col] = [0] * rows

    train_df = pd.DataFrame(synthetic_data)

    # Convert to format LIME expects (numpy array), handling categoricals
    # We need to encode strings to integers for LIME's internal math
    
    categorical_features_indices = []
    categorical_names = {}
    
    # Copy for encoding
    train_encoded = train_df.copy()

    for i, col in enumerate(_feature_names):
        if col in _cat_options or col == 'country':
            categorical_features_indices.append(i)
            # Create a mapping for this column
            if col in _cat_options:
                unique_vals = _cat_options[col]
            else:
                unique_vals = list(train_df[col].unique())
            
            categorical_names[i] = unique_vals
            
            # Encode the synthetic data column to match indices
            # Create a simple map: value -> index
            val_to_idx = {v: k for k, v in enumerate(unique_vals)}
            # Map values, default to 0 if not found
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


# --- 4. APP UI ---
st.title("🏨 Hotel Cancellation Predictor")

if model is None:
    st.warning("Please upload the required files (pkl, txt, csv) to the app directory.")
    st.stop()

# Form Inputs
user_input = {}

# Layout Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📝 Booking Info", "👤 Guest Profile", "⚠️ History & Risk", "📊 Scores"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        user_input['market_segment'] = st.selectbox("Market Segment", options=cat_options.get('market_segment', []))
        user_input['reserved_room_type'] = st.selectbox("Reserved Room Type", options=cat_options.get('reserved_room_type', []))
        user_input['deposit_type'] = st.selectbox("Deposit Type", options=cat_options.get('deposit_type', []))
    with col2:
        user_input['total_of_special_requests'] = st.number_input("Special Requests", min_value=0, max_value=10, value=0)
        user_input['required_car_parking_spaces'] = st.number_input("Car Parking Spaces", min_value=0, max_value=10, value=0)
        user_input['days_in_waiting_list'] = st.number_input("Days in Waiting List", 
                                                             min_value=int(num_ranges['days_in_waiting_list'][0]), 
                                                             max_value=int(num_ranges['days_in_waiting_list'][1]), value=0)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        user_input['customer_type'] = st.selectbox("Customer Type", options=cat_options.get('customer_type', []))
        user_input['country_group'] = st.selectbox("Country Group", options=cat_options.get('country_group', []))
    with col2:
        # Country isn't in your unique_values.csv, providing text input or common default
        user_input['country'] = st.text_input("Country Code (e.g., PRT, GBR)", value="PRT")

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        # Using selectbox for binary features is often clearer than 0/1 number input
        user_input['is_repeat_canceler'] = st.selectbox("Is Repeat Canceler?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        user_input['is_high_risk'] = st.selectbox("Is High Risk?", options=[0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    with col2:
        user_input['previous_cancellations'] = st.number_input("Previous Cancellations", min_value=0, value=0)
        user_input['booking_changes'] = st.number_input("Booking Changes", min_value=0, value=0)

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        mn, mx = num_ranges.get('commitment_score', (0.0, 10.0))
        user_input['commitment_score'] = st.slider("Commitment Score", float(mn), float(mx), float(mn))
    with col2:
        mn, mx = num_ranges.get('booking_stability', (-5.0, 5.0))
        user_input['booking_stability'] = st.slider("Booking Stability", float(mn), float(mx), float(mn))

# --- 5. PREDICTION ---
st.markdown("---")
predict_btn = st.button("🚀 Predict Cancellation", type="primary", use_container_width=True)

if predict_btn:
    # 1. Prepare Input DataFrame
    input_df = pd.DataFrame([user_input])
    
    # Ensure column order matches exactly what the model expects
    try:
        input_df = input_df[feature_names]
        
        # 2. Prediction
        pred_class = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]
        
        # 3. Results Display
        st.subheader("Results")
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if pred_class == 1:
                st.error("### Prediction: Cancelled")
                st.write("This booking is at risk of cancellation.")
            else:
                st.success("### Prediction: Not Cancelled")
                st.write("This booking is likely to be fulfilled.")
        
        with col_res2:
            st.metric("Cancellation Probability", f"{pred_proba[1]:.2%}")
            st.progress(float(pred_proba[1]))

        # 4. LIME Explanation
        st.markdown("---")
        st.subheader("🔍 LIME Explanation")
        st.write("Which features contributed most to this prediction?")
        
        with st.spinner("Calculating feature importance..."):
            
            # Wrapper Function: LIME passes a Numpy Array -> We need to convert back to DataFrame for Model Pipeline
            def predict_fn_lime(numpy_array):
                # numpy_array contains floats. Categoricals are indices.
                temp_df = pd.DataFrame(numpy_array, columns=feature_names)
                
                # Convert numeric indices back to original strings for categorical columns
                for col_idx, col_name in enumerate(feature_names):
                    if col_idx in lime_cat_names:
                        mapping_list = lime_cat_names[col_idx]
                        # Map index to string, safely handling out of bounds
                        temp_df[col_name] = temp_df[col_name].apply(
                            lambda x: mapping_list[int(x)] if 0 <= int(x) < len(mapping_list) else mapping_list[0]
                        )
                
                # Now predict using the original pipeline
                return model.predict_proba(temp_df)

            # Transform user input to the encoded format LIME expects
            lime_input_row = []
            for i, col in enumerate(feature_names):
                val = user_input[col]
                if i in lime_cat_names:
                    # Find index
                    try:
                        idx = lime_cat_names[i].index(val)
                    except ValueError:
                        # Fallback for unseen values (like a new Country code)
                        idx = 0 
                    lime_input_row.append(idx)
                else:
                    lime_input_row.append(val)
            
            lime_input_np = np.array(lime_input_row)

            # Explain
            exp = explainer.explain_instance(
                data_row=lime_input_np, 
                predict_fn=predict_fn_lime,
                num_features=10
            )
            
            # Display Plot
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Debug info - Feature Names Required:", feature_names)