#!/usr/bin/env python3
"""
Airborne Virus Collection Predictor - Streamlit App

A user-friendly web interface for predicting optimal virus collection parameters.
Users can either input parameters manually or upload a CSV file for batch predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import sys
import os
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

class VirusCollectionPredictor:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.model_info = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the saved model and related components."""
        models_dir = Path(__file__).parent.parent / 'models'
        
        # Load the trained model
        model_path = models_dir / 'best_model_voting_soft.joblib'
        if not model_path.exists():
            st.error(f"Model not found at {model_path}. Please run save_best_model.py first.")
            st.stop()
        
        self.model = joblib.load(model_path)
        
        # Load the label encoder
        encoder_path = models_dir / 'label_encoder.joblib'
        self.label_encoder = joblib.load(encoder_path)
        
        # Load model info
        info_path = models_dir / 'model_info.json'
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        self.feature_names = self.model_info['feature_names']
    
    def predict_single(self, RH, Temperature, Particles, Mod_Temp, Nozzle_temp, collection_medium=1):
        """Make prediction for a single set of parameters."""
        # Calculate temperature difference
        temp_diff = Nozzle_temp - Mod_Temp
        
        # Create feature vector
        features = [RH, Temperature, Particles, Mod_Temp, Nozzle_temp, temp_diff, collection_medium]
        feature_df = pd.DataFrame([features], columns=self.feature_names)
        
        # Make prediction
        prediction_encoded = self.model.predict(feature_df)[0]
        probabilities = self.model.predict_proba(feature_df)[0]
        
        # Decode prediction
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Create probability dictionary
        prob_dict = {class_name: prob for class_name, prob in zip(self.label_encoder.classes_, probabilities)}
        
        return prediction, prob_dict
    
    def predict_batch(self, df):
        """Make predictions for a batch of data."""
        results = []
        
        for _, row in df.iterrows():
            prediction, probabilities = self.predict_single(
                row['RH'], row['Temperature'], row['Particles'],
                row['Mod_Temp'], row['Nozzle_temp'], 
                row.get('collection_medium', 1)
            )
            
            result = {
                'RH': row['RH'],
                'Temperature': row['Temperature'],
                'Particles': row['Particles'],
                'Mod_Temp': row['Mod_Temp'],
                'Nozzle_temp': row['Nozzle_temp'],
                'Collection_Medium': row.get('collection_medium', 1),
                'Temp_Diff': row['Nozzle_temp'] - row['Mod_Temp'],
                'Prediction': prediction,
                **{f'Prob_{k}': v for k, v in probabilities.items()}
            }
            results.append(result)
        
        return pd.DataFrame(results)

def main():
    # Configure page
    st.set_page_config(
        page_title="Airborne Virus Collection Predictor",
        page_icon="🦠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    .optimal { background-color: #d4edda; border-color: #c3e6cb; }
    .excessive { background-color: #fff3cd; border-color: #ffeaa7; }
    .insufficient { background-color: #f8d7da; border-color: #f5c6cb; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🦠 Airborne Virus Collection Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered optimization for virus collection efficiency</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        with st.spinner('Loading AI model...'):
            st.session_state.predictor = VirusCollectionPredictor()
    
    predictor = st.session_state.predictor
    
    # Model info sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"""
        **Model Type:** {predictor.model_info['model_type']}
        
        **Expected Accuracy:** {predictor.model_info['expected_accuracy']:.1%}
        
        **Base Estimators:** {len(predictor.model_info['base_estimators'])}
        - Random Forest
        - Gradient Boosting  
        - Extra Trees
        - Logistic Regression
        - K-Nearest Neighbors
        
        **Target Classes:**
        - 🟢 **Optimal** (1.45-1.90 mL)
        - 🟡 **Excessive** (>1.90 mL)
        - 🔴 **Insufficient** (<1.45 mL)
        """)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Upload", "🔍 Parameter Explorer"])
    
    with tab1:
        st.header("Manual Parameter Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Environmental Conditions")
            RH = st.slider("Relative Humidity (%)", min_value=20.0, max_value=60.0, value=40.0, step=0.1)
            Temperature = st.slider("Temperature (°C)", min_value=20.0, max_value=30.0, value=24.0, step=0.1)
            Particles = st.number_input("Particle Count", min_value=100, max_value=10000, value=2500, step=100)
        
        with col2:
            st.subheader("Operational Parameters")
            Mod_Temp = st.slider("Moderator Temperature (°C)", min_value=15.0, max_value=35.0, value=26.0, step=0.1)
            Nozzle_temp = st.slider("Nozzle Temperature (°C)", min_value=20.0, max_value=45.0, value=33.0, step=0.1)
            
            st.subheader("Collection Medium")
            collection_medium_option = st.selectbox("Collection Medium", ["DI water", "AVL"], index=1)
            collection_medium = 0 if collection_medium_option == "DI water" else 1
            
            # Validation
            temp_diff = Nozzle_temp - Mod_Temp
            if temp_diff <= 0:
                st.warning("⚠️ Nozzle temperature should be higher than moderator temperature!")
            else:
                st.success(f"✅ Temperature difference: {temp_diff:.1f}°C")
        
        if st.button("🚀 Predict Collection Efficiency", type="primary", use_container_width=True):
            if temp_diff > 0:
                with st.spinner('Making prediction...'):
                    prediction, probabilities = predictor.predict_single(
                        RH, Temperature, Particles, Mod_Temp, Nozzle_temp, collection_medium
                    )
                
                # Display results
                display_single_prediction(prediction, probabilities, {
                    'RH': RH, 'Temperature': Temperature, 'Particles': Particles,
                    'Mod_Temp': Mod_Temp, 'Nozzle_temp': Nozzle_temp, 'Temp_Diff': temp_diff,
                    'Collection_Medium': collection_medium_option
                })
            else:
                st.error("Please ensure nozzle temperature is higher than moderator temperature.")
    
    with tab2:
        st.header("Batch Prediction from CSV")
        
        st.markdown("""
        Upload a CSV file with the following columns:
        - `RH` (Relative Humidity %)
        - `Temperature` (°C)
        - `Particles` (particle count)
        - `Mod_Temp` (Moderator Temperature °C)
        - `Nozzle_temp` (Nozzle Temperature °C)
        - `collection_medium` (optional, defaults to 1)
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                # Validate columns
                required_cols = ['RH', 'Temperature', 'Particles', 'Mod_Temp', 'Nozzle_temp']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    st.success(f"✅ File loaded successfully! {len(df)} rows found.")
                    
                    # Show preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head())
                    
                    if st.button("🎯 Run Batch Predictions", type="primary"):
                        with st.spinner('Processing predictions...'):
                            results = predictor.predict_batch(df)
                        
                        display_batch_results(results)
                        
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    with tab3:
        st.header("Parameter Space Explorer")
        
        st.markdown("Explore optimal operational parameters under specific environmental conditions. All combinations automatically ensure Nozzle temperature > Moderator temperature.")
        
        # Input method selection
        input_method = st.radio("Input Method:", ["Use Sliders", "Manual Input"], horizontal=True)
        
        col1, col2 = st.columns(2)
        
        if input_method == "Use Sliders":
            with col1:
                st.subheader("Environmental Conditions (Fixed)")
                explore_RH = st.select_slider("Relative Humidity (%)", options=[35, 37, 39, 41, 43, 45], value=41)
                explore_temp = st.select_slider("Temperature (°C)", options=[23, 24, 25, 26, 27], value=24)
                explore_particles = st.select_slider("Particles", options=list(range(1000, 5100, 100)), value=3000)
                
                st.subheader("Collection Medium")
                explore_collection_medium_option = st.selectbox("Collection Medium", ["DI water", "AVL"], index=1, key="explore_medium_slider")
                explore_collection_medium = 0 if explore_collection_medium_option == "DI water" else 1
            
            with col2:
                st.subheader("Operational Parameter Exploration")
                st.markdown("*Moderator and Nozzle temperatures will be varied automatically*")
                mod_temp_min = st.slider("Moderator Temp Min (°C)", 15.0, 35.0, 20.0, 0.5)
                mod_temp_max = st.slider("Moderator Temp Max (°C)", 15.0, 35.0, 32.0, 0.5)
                nozzle_temp_min = st.slider("Nozzle Temp Min (°C)", 20.0, 45.0, 25.0, 0.5)
                nozzle_temp_max = st.slider("Nozzle Temp Max (°C)", 20.0, 45.0, 40.0, 0.5)
                n_combinations = st.slider("Number of combinations", 20, 200, 50)
        else:
            with col1:
                st.subheader("Environmental Conditions (Fixed)")
                explore_RH = st.number_input("Relative Humidity (%)", min_value=20.0, max_value=60.0, value=41.0, step=0.5)
                explore_temp = st.number_input("Temperature (°C)", min_value=20.0, max_value=30.0, value=24.0, step=0.1)
                explore_particles = st.number_input("Particles", min_value=100, max_value=10000, value=3000, step=100)
                
                st.subheader("Collection Medium")
                explore_collection_medium_option = st.selectbox("Collection Medium", ["DI water", "AVL"], index=1, key="explore_medium_manual")
                explore_collection_medium = 0 if explore_collection_medium_option == "DI water" else 1
            
            with col2:
                st.subheader("Operational Parameter Exploration")
                st.markdown("*Define ranges for automatic variation*")
                mod_temp_min = st.number_input("Moderator Temp Min (°C)", min_value=15.0, max_value=35.0, value=20.0, step=0.5)
                mod_temp_max = st.number_input("Moderator Temp Max (°C)", min_value=15.0, max_value=35.0, value=32.0, step=0.5)
                nozzle_temp_min = st.number_input("Nozzle Temp Min (°C)", min_value=20.0, max_value=45.0, value=25.0, step=0.5)
                nozzle_temp_max = st.number_input("Nozzle Temp Max (°C)", min_value=20.0, max_value=45.0, value=40.0, step=0.5)
                n_combinations = st.number_input("Number of combinations", min_value=20, max_value=200, value=50, step=10)
        
        # Validation
        if mod_temp_min >= mod_temp_max:
            st.error("❌ Moderator Temp Max must be greater than Min")
        elif nozzle_temp_min >= nozzle_temp_max:
            st.error("❌ Nozzle Temp Max must be greater than Min")
        else:
            st.success("✅ Parameter ranges are valid - combinations will ensure Nozzle > Moderator")
            
        # Create ranges for the exploration function
        mod_temp_range = (mod_temp_min, mod_temp_max)
        nozzle_temp_range = (nozzle_temp_min, nozzle_temp_max)
        
        if st.button("🔍 Explore Parameter Space", type="primary"):
            with st.spinner('Generating parameter combinations...'):
                results = generate_parameter_exploration(
                    predictor, explore_RH, explore_temp, explore_particles, 
                    n_combinations, mod_temp_range, nozzle_temp_range, explore_collection_medium
                )
            
            display_exploration_results(results)

def display_single_prediction(prediction, probabilities, params):
    """Display results for single prediction with nice formatting."""
    
    # Main prediction card
    prediction_class = prediction.lower()
    st.markdown(f"""
    <div class="prediction-card {prediction_class}">
        <h2>🎯 Prediction Result</h2>
        <h1>{prediction}</h1>
        <p>Collection Volume Category</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability breakdown
    col1, col2, col3 = st.columns(3)
    
    colors = {'Optimal': '#28a745', 'Excessive': '#ffc107', 'Insufficient': '#dc3545'}
    
    for i, (class_name, prob) in enumerate(probabilities.items()):
        with [col1, col2, col3][i]:
            color = colors.get(class_name, '#6c757d')
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="color: {color}; margin: 0;">{class_name}</h3>
                <h2 style="margin: 0.5rem 0;">{prob:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    # Probability chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(probabilities.keys()),
            y=list(probabilities.values()),
            marker_color=[colors[k] for k in probabilities.keys()],
            text=[f'{v:.1%}' for v in probabilities.values()],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        yaxis_title="Probability",
        xaxis_title="Collection Category",
        showlegend=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Parameter summary
    st.subheader("📋 Input Parameters")
    param_df = pd.DataFrame([params])
    st.dataframe(param_df, use_container_width=True)

def display_batch_results(results):
    """Display results for batch predictions."""
    
    st.subheader("📊 Batch Prediction Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total = len(results)
    optimal_count = len(results[results['Prediction'] == 'Optimal'])
    excessive_count = len(results[results['Prediction'] == 'Excessive'])
    insufficient_count = len(results[results['Prediction'] == 'Insufficient'])
    
    with col1:
        st.metric("Total Predictions", total)
    with col2:
        st.metric("Optimal", optimal_count, f"{optimal_count/total:.1%}")
    with col3:
        st.metric("Excessive", excessive_count, f"{excessive_count/total:.1%}")
    with col4:
        st.metric("Insufficient", insufficient_count, f"{insufficient_count/total:.1%}")
    
    # Distribution chart
    prediction_counts = results['Prediction'].value_counts()
    fig = px.pie(
        values=prediction_counts.values,
        names=prediction_counts.index,
        title="Prediction Distribution",
        color_discrete_map={'Optimal': '#28a745', 'Excessive': '#ffc107', 'Insufficient': '#dc3545'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table
    st.subheader("🔍 Detailed Results")
    
    # Sort by optimal probability
    results_sorted = results.sort_values('Prob_Optimal', ascending=False)
    
    # Style the dataframe
    def style_prediction(val):
        if val == 'Optimal':
            return 'background-color: #d4edda'
        elif val == 'Excessive':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = results_sorted.style.applymap(style_prediction, subset=['Prediction'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Download link
    csv = results_sorted.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="virus_collection_predictions.csv",
        mime="text/csv"
    )

def display_exploration_results(results):
    """Display parameter exploration results."""
    
    st.subheader("🔍 Parameter Exploration Results")
    
    # Summary statistics in a more prominent layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        optimal_rate = len(results[results['Prediction'] == 'Optimal']) / len(results)
        st.metric("Optimal Rate", f"{optimal_rate:.1%}")
    
    with col2:
        avg_optimal_prob = results['Prob_Optimal'].mean()
        st.metric("Avg Optimal Probability", f"{avg_optimal_prob:.1%}")
    
    with col3:
        best_combo = results.loc[results['Prob_Optimal'].idxmax()]
        st.metric("Best Optimal Probability", f"{best_combo['Prob_Optimal']:.1%}")
    
    with col4:
        st.metric("Total Combinations", len(results))
    
    # Environmental conditions and best combination
    collection_medium_name = "DI water" if results['Collection_Medium'].iloc[0] == 0 else "AVL"
    st.info(f"🌡️ **Environmental Conditions:** RH: {results['RH'].iloc[0]:.1f}%, Temp: {results['Temperature'].iloc[0]:.1f}°C, Particles: {results['Particles'].iloc[0]:,}, Medium: {collection_medium_name}")
    st.success(f"🏆 **Best Operational Settings:** Mod: {best_combo['Mod_Temp']:.1f}°C, Nozzle: {best_combo['Nozzle_temp']:.1f}°C, Temp Diff: {best_combo['Temp_Diff']:.1f}°C")
    
    # Create larger visualizations
    st.subheader("📊 Interactive 3D Parameter Space")
    
    # 3D scatter plot with larger size and better formatting
    fig = px.scatter_3d(
        results,
        x='Mod_Temp',
        y='Nozzle_temp', 
        z='Temp_Diff',
        color='Prob_Optimal',
        size='Prob_Optimal',
        hover_data=['RH', 'Temperature', 'Particles', 'Collection_Medium', 'Prediction'],
        title=f'Operational Parameter Space (RH: {results["RH"].iloc[0]:.1f}%, Temp: {results["Temperature"].iloc[0]:.1f}°C, Particles: {results["Particles"].iloc[0]:,})',
        labels={
            'Mod_Temp': 'Moderator Temperature (°C)', 
            'Nozzle_temp': 'Nozzle Temperature (°C)', 
            'Temp_Diff': 'Temperature Difference (°C)',
            'Prob_Optimal': 'Optimal Probability'
        },
        color_continuous_scale='Viridis',
        height=700  # Make it taller
    )
    
    # Improve the layout and readability
    fig.update_traces(
        marker=dict(
            sizemin=8,
            sizeref=0.1,
            sizemode='diameter',
            line=dict(width=1, color='white')
        )
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Moderator Temperature (°C)',
            yaxis_title='Nozzle Temperature (°C)',
            zaxis_title='Temperature Difference (°C)',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        font=dict(size=12),
        title_x=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True, height=700)
    
    # Additional 2D projections for better analysis
    st.subheader("📈 2D Projections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mod Temp vs Nozzle Temp
        fig2d_1 = px.scatter(
            results,
            x='Mod_Temp',
            y='Nozzle_temp',
            color='Prob_Optimal',
            size='Prob_Optimal',
            title='Moderator vs Nozzle Temperature',
            labels={'Mod_Temp': 'Moderator Temp (°C)', 'Nozzle_temp': 'Nozzle Temp (°C)'},
            color_continuous_scale='Viridis',
            height=400
        )
        fig2d_1.update_traces(marker=dict(sizemin=6, sizeref=0.1, sizemode='diameter'))
        st.plotly_chart(fig2d_1, use_container_width=True)
    
    with col2:
        # Temperature Difference vs Optimal Probability
        fig2d_2 = px.scatter(
            results,
            x='Temp_Diff',
            y='Prob_Optimal',
            color='Prediction',
            title='Temperature Difference vs Optimal Probability',
            labels={'Temp_Diff': 'Temperature Difference (°C)', 'Prob_Optimal': 'Optimal Probability'},
            color_discrete_map={'Optimal': '#28a745', 'Excessive': '#ffc107', 'Insufficient': '#dc3545'},
            height=400
        )
        st.plotly_chart(fig2d_2, use_container_width=True)
    
    # Distribution analysis
    st.subheader("📊 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of Optimal Probabilities
        fig_hist = px.histogram(
            results,
            x='Prob_Optimal',
            nbins=20,
            title='Distribution of Optimal Probabilities',
            labels={'Prob_Optimal': 'Optimal Probability', 'count': 'Frequency'},
            color_discrete_sequence=['#1f77b4'],
            height=350
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot by prediction category
        fig_box = px.box(
            results,
            x='Prediction',
            y='Prob_Optimal',
            title='Optimal Probability by Prediction Category',
            color='Prediction',
            color_discrete_map={'Optimal': '#28a745', 'Excessive': '#ffc107', 'Insufficient': '#dc3545'},
            height=350
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Top 10 combinations with enhanced display
    st.subheader("🏆 Top 10 Combinations")
    top_10 = results.nlargest(10, 'Prob_Optimal')[
        ['RH', 'Temperature', 'Particles', 'Mod_Temp', 'Nozzle_temp', 'Collection_Medium', 'Temp_Diff', 'Prediction', 'Prob_Optimal']
    ].round(2)
    
    # Style the dataframe
    def highlight_optimal(val):
        if isinstance(val, str) and val == 'Optimal':
            return 'background-color: #d4edda'
        elif isinstance(val, str) and val == 'Excessive':
            return 'background-color: #fff3cd'
        elif isinstance(val, str) and val == 'Insufficient':
            return 'background-color: #f8d7da'
        return ''
    
    styled_top_10 = top_10.style.applymap(highlight_optimal, subset=['Prediction'])
    st.dataframe(styled_top_10, use_container_width=True)
    
    # Download option
    csv = results.round(3).to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=csv,
        file_name=f"parameter_exploration_results_{len(results)}_combinations.csv",
        mime="text/csv"
    )

def generate_parameter_exploration(predictor, rh_value, temp_value, particles, n_combinations, 
                                  mod_temp_range, nozzle_temp_range, collection_medium=1):
    """Generate parameter combinations for exploration with fixed environmental conditions.
    
    Ensures all combinations satisfy Nozzle_temp > Mod_temp constraint by intelligent sampling.
    """
    np.random.seed(42)
    
    results = []
    
    for _ in range(n_combinations):
        # Fixed environmental conditions
        RH = rh_value
        Temperature = temp_value
        
        # Smart generation to ensure Nozzle > Moderator constraint
        # Strategy: Generate moderator temp first, then nozzle temp > moderator
        
        Mod_Temp = np.random.uniform(mod_temp_range[0], mod_temp_range[1])
        
        # Calculate valid nozzle temperature range
        # Nozzle must be > Mod_Temp and within the specified nozzle range
        effective_nozzle_min = max(Mod_Temp + 0.5, nozzle_temp_range[0])  # At least 0.5°C higher
        effective_nozzle_max = nozzle_temp_range[1]
        
        # If no valid nozzle range exists, adjust moderator temperature
        if effective_nozzle_min > effective_nozzle_max:
            # Adjust moderator to ensure valid nozzle range exists
            max_valid_mod = nozzle_temp_range[1] - 0.5
            Mod_Temp = min(Mod_Temp, max_valid_mod)
            Mod_Temp = max(Mod_Temp, mod_temp_range[0])  # Still within mod range
            
            # Recalculate nozzle range
            effective_nozzle_min = max(Mod_Temp + 0.5, nozzle_temp_range[0])
            effective_nozzle_max = nozzle_temp_range[1]
        
        # Generate nozzle temperature within valid range
        if effective_nozzle_min <= effective_nozzle_max:
            Nozzle_temp = np.random.uniform(effective_nozzle_min, effective_nozzle_max)
        else:
            # Fallback: set nozzle to be moderator + 1°C
            Nozzle_temp = Mod_Temp + 1.0
        
        # Make prediction
        prediction, probabilities = predictor.predict_single(
            RH, Temperature, particles, Mod_Temp, Nozzle_temp, collection_medium
        )
        
        result = {
            'RH': round(RH, 1),
            'Temperature': round(Temperature, 1),
            'Particles': particles,
            'Mod_Temp': round(Mod_Temp, 1),
            'Nozzle_temp': round(Nozzle_temp, 1),
            'Collection_Medium': collection_medium,
            'Temp_Diff': round(Nozzle_temp - Mod_Temp, 1),
            'Prediction': prediction,
            **{f'Prob_{k}': v for k, v in probabilities.items()}
        }
        results.append(result)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    main()