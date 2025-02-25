import streamlit as st
import numpy as np
from itertools import product
import joblib
import os
import pandas as pd

def find_optimal_parameters(reg_model, class_model, RH, Temperature, Particles, medium):
    # Create grid of nozzle and mod_temp values
    nozzle_range = np.arange(10, 35, 0.5)
    mod_temp_range = np.arange(10, 35, 0.5)
    
    # Create all possible combinations
    combinations = list(product(nozzle_range, mod_temp_range))
    
    # Create input array for each combination
    input_arrays = []
    for nozzle, mod_temp in combinations:
        delta_t = nozzle - mod_temp
        input_arrays.append([Temperature, RH, Particles, mod_temp, nozzle, delta_t, medium])
    
    # Convert to numpy array
    X = np.array(input_arrays)
    
    # Get predictions
    vol_predictions = reg_model.predict(X)
    class_predictions = class_model.predict(X)
    
    # Filter valid combinations
    valid_mask = (class_predictions == 1) & (vol_predictions >= 1.45) & (vol_predictions <= 1.76)
    
    if not any(valid_mask):
        return None, None, None
    
    # Get valid combinations and their predicted volumes
    valid_combinations = np.array(combinations)[valid_mask]
    valid_volumes = vol_predictions[valid_mask]
    
    # Find the combination closest to the target volume
    target = 1.605
    best_idx = np.argmin(np.abs(valid_volumes - target))
    
    best_nozzle, best_mod_temp = valid_combinations[best_idx]
    best_volume = valid_volumes[best_idx]
    
    return best_nozzle, best_mod_temp, best_volume

def process_csv(df, reg_model, class_model):
    results = []
    for _, row in df.iterrows():
        best_nozzle, best_mod_temp, best_volume = find_optimal_parameters(
            reg_model, 
            class_model, 
            row['RH'], 
            row['Temperature'], 
            row['Particles'], 
            row['medium']
        )
        
        results.append({
            'Temperature': row['Temperature'],
            'RH': row['RH'],
            'Particles': row['Particles'],
            'medium': row['medium'],
            'Optimal_Nozzle_Temp': best_nozzle,
            'Optimal_Mod_Temp': best_mod_temp,
            'Predicted_Volume': best_volume,
            'Delta_T': best_nozzle - best_mod_temp if best_nozzle is not None else None
        })
    
    return pd.DataFrame(results)

def main():
    st.title("Temperature Optimizer")
    
    # Check if models directory exists
    if not os.path.exists('report_2.11/models'):
        st.error("Models directory not found. Please ensure 'report_2.11/models' directory exists.")
        st.stop()

    # Load models
    try:
        reg_model = joblib.load('models/best_regression_model.joblib')
        class_model = joblib.load('models/best_classification_model.joblib')
    except Exception as e:
        st.error(f"Could not load models. Error: {str(e)}")
        st.error("Please ensure model files exist in 'models' directory.")
        st.stop()

    # Add tabs for single input and CSV upload
    tab1, tab2 = st.tabs(["Single Input", "CSV Upload"])

    with tab1:
        # Original single input form
        with st.form("input_form"):
            temperature = st.number_input(
                "Room Temperature (°C)", 
                min_value=15.0, 
                max_value=35.0, 
                value=23.0,
                help="Enter room temperature between 15°C and 35°C"
            )
            
            rh = st.number_input(
                "Relative Humidity (%)", 
                min_value=20.0, 
                max_value=80.0, 
                value=45.0,
                help="Enter relative humidity between 20% and 80%"
            )
            
            particles = st.number_input(
                "Particle Count", 
                min_value=0, 
                max_value=10000, 
                value=2000,
                help="Enter particle count between 0 and 5000"
            )
            
            medium = st.selectbox(
                "Medium Type",
                options=[0, 1],
                format_func=lambda x: "Type 0" if x == 0 else "Type 1",
                help="Select medium type (0 or 1)"
            )
            
            submitted = st.form_submit_button("Find Optimal Temperatures")

        if submitted:
            with st.spinner("Calculating optimal temperatures..."):
                best_nozzle, best_mod_temp, best_volume = find_optimal_parameters(
                    reg_model, class_model, rh, temperature, particles, medium
                )
                
                if best_nozzle is None:
                    st.error("No valid solution found for these conditions.")
                else:
                    st.success("Optimal parameters found!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Optimal Nozzle Temperature", f"{best_nozzle:.1f}°C")
                        st.metric("Optimal Moderator Temperature", f"{best_mod_temp:.1f}°C")
                    with col2:
                        st.metric("Predicted Volume", f"{best_volume:.3f} mL")
                        st.metric("Temperature Difference", f"{best_nozzle - best_mod_temp:.1f}°C")

    with tab2:
        st.write("Upload a CSV file with columns: Temperature, RH, Particles, medium")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                required_columns = ['Temperature', 'RH', 'Particles', 'medium']
                
                # Verify all required columns are present
                if not all(col in input_df.columns for col in required_columns):
                    st.error(f"CSV must contain all required columns: {', '.join(required_columns)}")
                    st.stop()
                
                # Verify data ranges
                if not (
                    (input_df['Temperature'].between(15, 35)).all() and
                    (input_df['RH'].between(20, 80)).all() and
                    (input_df['Particles'].between(0, 10000)).all() and
                    (input_df['medium'].isin([0, 1])).all()
                ):
                    st.error("Some values in the CSV are outside the acceptable ranges.")
                    st.stop()
                
                with st.spinner("Processing CSV data..."):
                    results_df = process_csv(input_df, reg_model, class_model)
                    
                    # Display results
                    st.write("Results:")
                    st.dataframe(results_df)
                    
                    # Add download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="optimization_results.csv",
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    main() 