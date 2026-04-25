# House Price Prediction - Machine Learning Project

A complete machine learning project for predicting house prices using Python, scikit-learn, and Flask with a web-based interface.

## 📋 Project Overview

This project implements a full ML pipeline for house price prediction:

1. **Data Loading & Preprocessing**: Handles missing values, categorical encoding, and feature scaling
2. **Exploratory Data Analysis (EDA)**: Generates correlation heatmaps and feature distribution plots
3. **Model Training**: Trains three regression models
4. **Model Evaluation**: Compares models using R² score and RMSE metrics
5. **Web Application**: Flask-based REST API with interactive HTML frontend

## 📁 Project Structure

```
House Price Prediction/
├── main.py                          # ML pipeline and model training
├── app.py                           # Flask web application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── House Price Prediction Dataset.csv  # Input dataset
├── eda_analysis.png                # EDA visualization (generated)
├── feature_importance.png          # Feature importance plots (generated)
├── models/                         # Trained models and artifacts
│   ├── best_model.pkl             # Best performing model
│   ├── scaler.pkl                 # Feature scaler
│   ├── label_encoders.pkl         # Categorical encoders
│   ├── feature_names.pkl          # Feature names
│   └── all_models.pkl             # All trained models
└── templates/
    └── index.html                  # Web interface
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Navigate to project directory**
```bash
cd "House Price Prediction"
```

2. **Create virtual environment (optional but recommended)**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### Step 1: Train the Model

Run the ML pipeline to train and evaluate models:

```bash
python main.py
```

**Output:**
- Trained models saved in `models/` directory
- EDA visualizations saved as PNG files
- Feature importance plots generated
- Model comparison results displayed in console

### Step 2: Start Web Application

Launch the Flask web app:

```bash
python app.py
```

**Output:**
```
Running on http://localhost:5000
Press CTRL+C to quit
```

### Step 3: Open Web Interface

1. Open your browser
2. Go to `http://localhost:5000`
3. Enter house features
4. Click "Predict Price"

## 📊 Dataset Format

The CSV file should contain the following columns:

| Column | Type | Description |
|--------|------|-------------|
| Id | Integer | Unique identifier |
| Area | Float | House area in square feet |
| Bedrooms | Integer | Number of bedrooms |
| Bathrooms | Float | Number of bathrooms |
| Floors | Integer | Number of floors |
| YearBuilt | Integer | Year the house was built |
| Location | String | Location type (Downtown, Urban, Suburban, Rural) |
| Condition | String | House condition (Excellent, Good, Fair, Poor) |
| Garage | String | Garage availability (Yes, No) |
| Price | Float | House price (target variable) |

## 🤖 Models Implemented

### 1. Linear Regression
- Simple linear model
- Fast training and prediction
- Interpretable coefficients

### 2. Decision Tree Regressor
- Tree-based model with max_depth=20
- Captures non-linear relationships
- Provides feature importance

### 3. Random Forest Regressor
- Ensemble of 100 decision trees
- Best performance typically
- Robust feature importance scores

## 📈 Evaluation Metrics

### R² Score
- Measures proportion of variance explained
- Range: 0 to 1 (higher is better)
- Formula: R² = 1 - (SS_res / SS_tot)

### RMSE (Root Mean Square Error)
- Measures prediction error magnitude
- Lower values indicate better predictions
- Formula: RMSE = √(MSE)

## 🌐 Web Application Features

### Input Fields
- **Area**: House area in square feet (100-10,000)
- **Bedrooms**: Number of bedrooms (1-10)
- **Bathrooms**: Number of bathrooms (1-10)
- **Floors**: Number of floors (1-5)
- **Year Built**: Construction year (1800-2024)
- **Location**: Downtown, Urban, Suburban, or Rural
- **Condition**: Excellent, Good, Fair, or Poor
- **Garage**: Yes or No

### Output
- Predicted house price
- Real-time calculation using trained model
- Formatted currency display

### API Endpoints

**POST /predict**
- Accepts house features in JSON format
- Returns predicted price
```json
{
  "area": 2500,
  "bedrooms": 3,
  "bathrooms": 2.5,
  "floors": 2,
  "yearBuilt": 2000,
  "location": "Suburban",
  "condition": "Good",
  "garage": "Yes"
}
```

**GET /info**
- Returns model information and valid input values
```json
{
  "model_name": "House Price Prediction Model",
  "version": "1.0",
  "features": ["Area", "Bedrooms", ...],
  "valid_locations": ["Downtown", "Urban", "Suburban", "Rural"],
  "valid_conditions": ["Excellent", "Good", "Fair", "Poor"],
  "valid_garage": ["Yes", "No"]
}
```

## 📊 Data Preprocessing Steps

### 1. Missing Value Handling
- Numerical columns: Filled with mean value
- Categorical columns: Filled with mode value

### 2. Categorical Encoding
- Label Encoding for categorical variables
- Separate encoder for each categorical column
- Encoders saved for prediction preprocessing

### 3. Feature Scaling
- StandardScaler applied to all features
- Scaler saved for prediction preprocessing
- Ensures features are on comparable scales

## 🎯 Feature Importance

Tree-based models provide feature importance scores:

- **Decision Tree Importance**: Shows split-based importance
- **Random Forest Importance**: Averaged across all trees
- Higher values = more important for predictions

Visualizations automatically generated and saved to `feature_importance.png`

## 💾 Model Artifacts

All trained objects are saved and loaded for predictions:

- **best_model.pkl**: The selected best-performing model
- **scaler.pkl**: StandardScaler for feature normalization
- **label_encoders.pkl**: Encoders for categorical variables
- **feature_names.pkl**: Feature column names in correct order
- **all_models.pkl**: All three trained models for comparison

## 🔧 Configuration

### Model Parameters

**Linear Regression**
```python
lr_model = LinearRegression()
```

**Decision Tree**
```python
dt_model = DecisionTreeRegressor(random_state=42, max_depth=20)
```

**Random Forest**
```python
rf_model = RandomForestRegressor(
    n_estimators=100, 
    random_state=42, 
    max_depth=20, 
    n_jobs=-1
)
```

### Train-Test Split
- Test size: 20%
- Training size: 80%
- Random state: 42

## 🐛 Troubleshooting

### Model Not Found
**Error**: "Model not loaded. Please run main.py first."
- **Solution**: Run `python main.py` to train the model

### Invalid Input
**Error**: "Invalid input" or "Missing required field"
- **Solution**: Check all form fields are filled correctly
- Verify Location, Condition, and Garage values are valid

### Connection Error
**Error**: "Failed to connect to server"
- **Solution**: 
  1. Ensure Flask app is running (`python app.py`)
  2. Check http://localhost:5000 is accessible
  3. Verify no other service is using port 5000

### Import Errors
**Error**: "ModuleNotFoundError: No module named 'flask'"
- **Solution**: Run `pip install -r requirements.txt`

## 📝 Code Comments

All code includes comprehensive comments explaining:
- Function purposes and arguments
- Data processing steps
- Model training procedure
- Prediction workflow

## 🔄 Workflow

```
┌─────────────────────────┐
│   Load CSV Dataset      │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Data Preprocessing     │
│ - Handle missing values │
│ - Encode categorical    │
│ - Scale features        │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Exploratory Analysis   │
│ - Statistics            │
│ - Correlation heatmap   │
│ - Distribution plots    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Train-Test Split       │
│ 80% training, 20% test  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Train 3 Models         │
│ - Linear Regression     │
│ - Decision Tree         │
│ - Random Forest         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Evaluate Models        │
│ - R² Score              │
│ - RMSE                  │
│ - Feature Importance    │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Save Best Model        │
│ - Model artifacts       │
│ - Preprocessors         │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Web Application Ready  │
│  Start Flask server     │
└─────────────────────────┘
```

## 📚 Python Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning models and metrics
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **flask**: Web framework
- **pickle**: Model serialization

## 🎓 Learning Outcomes

This project demonstrates:
- Complete ML pipeline implementation
- Data preprocessing techniques
- Model training and evaluation
- Feature importance analysis
- Web application development
- Model persistence and loading
- RESTful API design
- Frontend-backend integration

## 📄 License

This project is provided as-is for educational purposes.

## 🤝 Contributing

Feel free to enhance this project:
- Add more preprocessing techniques
- Implement additional models
- Improve UI/UX
- Add data validation
- Create unit tests
- Deploy to cloud platforms

## ✨ Future Enhancements

- [ ] Cross-validation for robust evaluation
- [ ] Hyperparameter tuning
- [ ] Model ensemble techniques
- [ ] Feature selection algorithms
- [ ] Prediction confidence intervals
- [ ] Data augmentation
- [ ] Model versioning
- [ ] Docker containerization
- [ ] Cloud deployment
- [ ] API documentation (Swagger)

---

**Happy Predicting! 🏠📊**
