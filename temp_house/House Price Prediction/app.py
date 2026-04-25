"""
House Price Prediction - Flask Web Application
Web interface for making real-time house price predictions.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# ============================================================================
# LOAD MODELS AND PREPROCESSING OBJECTS
# ============================================================================
def load_model_artifacts():
    """
    Load pre-trained model and preprocessing objects.
    
    Returns:
        tuple: (model, scaler, label_encoders, feature_names)
    """
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    try:
        # Load model
        with open(os.path.join(models_dir, 'best_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Load label encoders
        with open(os.path.join(models_dir, 'label_encoders.pkl'), 'rb') as f:
            label_encoders = pickle.load(f)
        
        # Load feature names
        with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        
        return model, scaler, label_encoders, feature_names
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run 'python main.py' first to train the model.")
        return None, None, None, None


# Load model artifacts when app starts
model, scaler, label_encoders, feature_names = load_model_artifacts()


# ============================================================================
# ROUTES
# ============================================================================
@app.route('/')
def index():
    """
    Render the home page with prediction form.
    
    Returns:
        str: Rendered index.html template
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict house price based on user input.
    
    Expected JSON payload:
    {
        "area": float,
        "bedrooms": int,
        "bathrooms": float,
        "floors": int,
        "yearBuilt": int,
        "location": str (Downtown, Urban, Suburban, Rural),
        "condition": str (Excellent, Good, Fair, Poor),
        "garage": str (Yes/No)
    }
    
    Returns:
        json: Prediction result or error message
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please run main.py first.'
            }), 500
        
        # Get input data
        data = request.json
        
        # Validate required fields
        required_fields = ['area', 'bedrooms', 'bathrooms', 'floors', 
                          'yearBuilt', 'location', 'condition', 'garage']
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Prepare feature vector
        features_dict = {}
        
        # Add numerical features
        features_dict['Area'] = float(data['area'])
        features_dict['Bedrooms'] = int(data['bedrooms'])
        features_dict['Bathrooms'] = float(data['bathrooms'])
        features_dict['Floors'] = int(data['floors'])
        features_dict['YearBuilt'] = int(data['yearBuilt'])
        
        # Encode categorical features
        features_dict['Location'] = label_encoders['Location'].transform(
            [data['location']])[0]
        features_dict['Condition'] = label_encoders['Condition'].transform(
            [data['condition']])[0]
        features_dict['Garage'] = label_encoders['Garage'].transform(
            [data['garage']])[0]
        
        # Create feature array in correct order
        feature_array = np.array([[
            features_dict[name] for name in feature_names
        ]])
        
        # Scale features
        scaled_features = scaler.transform(feature_array)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Prepare response
        return jsonify({
            'success': True,
            'predicted_price': round(prediction, 2),
            'formatted_price': f"${prediction:,.2f}",
            'message': 'Prediction successful!'
        }), 200
    
    except ValueError as e:
        return jsonify({
            'error': f'Invalid input: {str(e)}'
        }), 400
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/info', methods=['GET'])
def info():
    """
    Get information about the model and valid input ranges.
    
    Returns:
        json: Model information and input specifications
    """
    info_data = {
        'model_name': 'House Price Prediction Model',
        'version': '1.0',
        'features': list(feature_names) if feature_names else [],
        'valid_locations': ['Downtown', 'Urban', 'Suburban', 'Rural'],
        'valid_conditions': ['Excellent', 'Good', 'Fair', 'Poor'],
        'valid_garage': ['Yes', 'No']
    }
    
    return jsonify(info_data), 200


# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MAIN APPLICATION
# ============================================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - FLASK WEB APPLICATION")
    print("="*70)
    
    if model is None:
        print("\n❌ ERROR: Model artifacts not found!")
        print("Please run 'python main.py' first to train the model.")
        print("="*70 + "\n")
    else:
        print("\n✓ Model loaded successfully!")
        print("✓ Starting Flask development server...")
        print("\n📍 Open http://localhost:5000 in your browser")
        print("="*70 + "\n")
        
        # Run the Flask app
        app.run(debug=True, host='localhost', port=5000)
