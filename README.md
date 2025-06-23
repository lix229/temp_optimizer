# Airborne Virus Collection Predictor - Streamlit App

A user-friendly web interface for predicting optimal virus collection parameters using AI/ML models.

## Features

### 🎯 Single Prediction
- Manual input of environmental and operational parameters
- Real-time validation and feedback
- Interactive sliders for parameter adjustment
- Detailed prediction results with probability breakdown

### 📁 Batch Upload
- CSV file upload for multiple predictions
- Automatic validation of required columns
- Batch processing with summary statistics
- Downloadable results

### 🔍 Parameter Explorer
- **Fixed Environmental Conditions**: Set specific RH, temperature, and particle count
- **Operational Parameter Exploration**: Vary moderator and nozzle temperatures within defined ranges
- **Dual Input Methods**: Sliders or manual numeric input for precise control
- **Enhanced Particle Range**: 100-step increments up to 10,000 particles
- **Large Interactive 3D Visualization**: 700px height showing operational parameter space
- **Multiple View Perspectives**: 3D, 2D projections, and distribution analysis
- **Comprehensive Analysis**: Histograms, box plots, and statistical summaries
- **Intelligent Constraint Handling**: Automatically generates valid combinations ensuring Nozzle > Moderator temperature, regardless of input ranges
- Top combination recommendations with downloadable results

## Installation & Setup

1. **Navigate to the streamlit_app directory:**
   ```bash
   cd streamlit_app
   ```

2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the trained model exists:**
   Make sure you have run the model training script from the parent directory:
   ```bash
   cd ..
   python save_best_model.py
   cd streamlit_app
   ```

## Running the Application

1. **Start the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser:**
   The app will automatically open at `http://localhost:8501`

## Usage Guide

### Manual Input Mode
1. Use the sliders to set environmental conditions:
   - Relative Humidity (20-60%)
   - Temperature (20-30°C)
   - Particle Count (100-10,000)

2. Set operational parameters:
   - Moderator Temperature (15-35°C)
   - Nozzle Temperature (20-45°C)
   - **Note:** Nozzle temperature must be higher than moderator temperature

3. Click "Predict Collection Efficiency" to get results

### CSV Upload Mode
1. Prepare a CSV file with the following columns:
   - `RH`: Relative Humidity (%)
   - `Temperature`: Temperature (°C)
   - `Particles`: Particle count
   - `Mod_Temp`: Moderator Temperature (°C)
   - `Nozzle_temp`: Nozzle Temperature (°C)
   - `collection_medium`: (optional, defaults to 1)

2. Upload the file using the file uploader
3. Review the data preview
4. Click "Run Batch Predictions"
5. Download results as CSV

### Parameter Explorer Mode
1. **Choose Input Method**: Select "Use Sliders" or "Manual Input"
2. **Set Environmental Conditions (Fixed)**:
   - **RH**: Single value for relative humidity (35-45%)
   - **Temperature**: Single value for temperature (23-27°C)
   - **Particles**: Single value for particle count (100-10,000)
3. **Set Operational Parameter Ranges**:
   - **Moderator Temperature**: Define min/max range for automatic variation
   - **Nozzle Temperature**: Define min/max range for automatic variation
   - **Combinations**: Generate 20-200 parameter combinations
4. **Analyze Results**:
   - Fixed environmental conditions with varied operational parameters
   - Interactive 3D visualization (700px height) showing operational space
   - 2D projections for detailed analysis
   - Distribution histograms and box plots
   - Top 10 combinations table
   - Download full results as CSV

## Model Information

- **Model Type:** Voting Classifier (Soft)
- **Expected Accuracy:** 72.0%
- **Base Estimators:** 5 (Random Forest, Gradient Boosting, Extra Trees, Logistic Regression, K-Nearest Neighbors)

### Target Classes
- **🟢 Optimal** (1.45-1.90 mL): Ideal collection volume
- **🟡 Excessive** (>1.90 mL): Too much collection
- **🔴 Insufficient** (<1.45 mL): Too little collection

## Troubleshooting

### Common Issues

1. **Model not found error:**
   - Ensure you've run `python save_best_model.py` from the parent directory
   - Check that the `models/` directory exists with the required files

2. **Import errors:**
   - Make sure all requirements are installed: `pip install -r requirements.txt`
   - Verify you're running from the correct directory

3. **CSV upload issues:**
   - Ensure your CSV has the required column names (case-sensitive)
   - Check that numeric columns contain valid numbers
   - Verify Nozzle_temp > Mod_Temp for all rows

### File Structure
```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file

../models/              # Required model files (from parent directory)
├── best_model_voting_soft.joblib
├── label_encoder.joblib
└── model_info.json
```

## Performance Tips

- For large CSV files (>1000 rows), consider breaking them into smaller batches
- The Parameter Explorer works best with 20-50 combinations for responsive interaction
- Use the single prediction mode for real-time parameter optimization

## Support

For issues or questions:
1. Check that all dependencies are installed correctly
2. Verify the model files exist in the parent directory
3. Ensure CSV files follow the required format
4. Check the Streamlit console for detailed error messages