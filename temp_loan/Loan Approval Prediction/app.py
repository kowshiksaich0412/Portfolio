"""
Loan Approval Prediction Flask Web Application
Provides REST API and web interface for making loan approval predictions.
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================
app = Flask(__name__, 
           template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
           static_folder=os.path.join(os.path.dirname(__file__), 'static'))

app.config['SECRET_KEY'] = 'loan-approval-secret-key-2026'

# Model paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'label_encoders.pkl')

# Global model and preprocessing objects
model = None
scaler = None
label_encoders = None


# ============================================================================
# Model Loading
# ============================================================================
def load_model_and_preprocessing():
    """Load the trained model and preprocessing objects."""
    global model, scaler, label_encoders
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler loaded successfully")
    except Exception as e:
        print(f"✗ Error loading scaler: {e}")
        return False
    
    try:
        with open(ENCODER_PATH, 'rb') as f:
            label_encoders = pickle.load(f)
        print(f"✓ Label encoders loaded successfully")
    except Exception as e:
        print(f"✗ Error loading encoders: {e}")
        return False
    
    return True


# ============================================================================
# Data Preprocessing for Prediction
# ============================================================================
def preprocess_input(data_dict):
    """
    Preprocess user input for model prediction.
    
    Args:
        data_dict: Dictionary containing user input
        
    Returns:
        Preprocessed feature array, or None if error
    """
    try:
        # Create DataFrame from input
        df = pd.DataFrame([data_dict])
        
        # Categorical and numerical columns
        categorical_cols = list(label_encoders.keys())
        numerical_cols = ['Age', 'Income', 'Credit_Score', 'Employment_Length', 'Loan_Amount', 'Interest_Rate']
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in df.columns:
                df[col] = label_encoders[col].transform([df[col].iloc[0]])
        
        # Ensure column order matches training data
        all_cols = numerical_cols + categorical_cols
        X = df[all_cols].values.astype(float)
        
        # Normalize numerical features
        # Create a full-sized array for scaler
        scaler_input = np.zeros((1, len(all_cols)))
        scaler_input[:, :len(numerical_cols)] = X[:, :len(numerical_cols)]
        scaler_input[:, len(numerical_cols):] = X[:, len(numerical_cols):]
        
        X_scaled = scaler.transform(X)
        
        return X_scaled
    
    except Exception as e:
        print(f"Error preprocessing input: {e}")
        return None


# ============================================================================
# Routes
# ============================================================================
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests.
    
    Expected JSON format:
    {
        "age": 35,
        "income": 75000,
        "credit_score": 720,
        "employment_length": 10,
        "loan_amount": 250000,
        "interest_rate": 4.5,
        "loan_type": "Personal",
        "employment_type": "Employed"
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        required_fields = ['age', 'income', 'credit_score', 'employment_length',
                          'loan_amount', 'interest_rate', 'loan_type', 'employment_type']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Prepare input dictionary with correct column names
        input_data = {
            'Age': float(data['age']),
            'Income': float(data['income']),
            'Credit_Score': float(data['credit_score']),
            'Employment_Length': float(data['employment_length']),
            'Loan_Amount': float(data['loan_amount']),
            'Interest_Rate': float(data['interest_rate']),
            'Loan_Type': str(data['loan_type']),
            'Employment_Type': str(data['employment_type'])
        }
        
        # Preprocess input
        X_scaled = preprocess_input(input_data)
        if X_scaled is None:
            return jsonify({'error': 'Error preprocessing input'}), 400
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        # Prepare response
        response = {
            'prediction': 'Approved' if prediction == 1 else 'Rejected',
            'approval_probability': float(probability[1]),
            'rejection_probability': float(probability[0]),
            'confidence': float(max(probability)) * 100,
            'status': 'success'
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}', 'status': 'error'}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoders_loaded': label_encoders is not None
    }), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == '__main__':
    print("="*80)
    print("LOAN APPROVAL PREDICTION - FLASK WEB APPLICATION")
    print("="*80)
    
    # Load model and preprocessing objects
    print("\nLoading model and preprocessing objects...")
    if not load_model_and_preprocessing():
        print("✗ Failed to load model. Please train the model first using train.py")
        exit(1)
    
    print("\n✓ Application initialized successfully!")
    print("✓ Starting Flask server...")
    print("\nAccess the application at: http://localhost:5000")
    print("\nAPI Endpoints:")
    print("  GET  /               - Web interface")
    print("  POST /predict        - Make predictions")
    print("  GET  /api/health     - Health check")
    print("\n" + "="*80)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
