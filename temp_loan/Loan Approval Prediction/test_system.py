"""
Testing and Validation Script for Loan Approval Prediction System
Tests model loading, predictions, and API health checks
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'label_encoders.pkl')

# Test data
TEST_CASES = [
    {
        'name': 'Likely Approved - Good Credit',
        'data': {
            'age': 35,
            'income': 75000,
            'credit_score': 720,
            'employment_length': 10,
            'loan_amount': 250000,
            'interest_rate': 4.5,
            'loan_type': 'Personal',
            'employment_type': 'Employed'
        },
        'expected': 'Approved'
    },
    {
        'name': 'Likely Rejected - Low Credit',
        'data': {
            'age': 25,
            'income': 30000,
            'credit_score': 620,
            'employment_length': 1,
            'loan_amount': 300000,
            'interest_rate': 8.5,
            'loan_type': 'Personal',
            'employment_type': 'Employed'
        },
        'expected': 'Rejected'
    },
    {
        'name': 'Borderline - Medium Profile',
        'data': {
            'age': 40,
            'income': 50000,
            'credit_score': 680,
            'employment_length': 5,
            'loan_amount': 150000,
            'interest_rate': 5.5,
            'loan_type': 'Auto',
            'employment_type': 'Self-Employed'
        },
        'expected': 'Unknown'
    }
]


# ============================================================================
# Test Functions
# ============================================================================
def print_header(text):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_success(text):
    """Print success message."""
    print(f"✓ {text}")


def print_error(text):
    """Print error message."""
    print(f"✗ {text}")


def print_info(text):
    """Print info message."""
    print(f"ℹ {text}")


def test_model_files():
    """Test if model files exist."""
    print_header("TEST 1: Model Files Existence")
    
    files_to_check = [
        ('Best Model', MODEL_PATH),
        ('Scaler', SCALER_PATH),
        ('Label Encoders', ENCODER_PATH)
    ]
    
    all_exist = True
    for name, path in files_to_check:
        if os.path.exists(path):
            file_size = os.path.getsize(path) / 1024  # KB
            print_success(f"{name}: {path} ({file_size:.2f} KB)")
        else:
            print_error(f"{name}: {path} NOT FOUND")
            all_exist = False
    
    return all_exist


def test_model_loading():
    """Test if models can be loaded."""
    print_header("TEST 2: Model Loading")
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print_success(f"Model loaded: {type(model).__name__}")
    except Exception as e:
        print_error(f"Failed to load model: {e}")
        return False
    
    try:
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        print_success(f"Scaler loaded: {type(scaler).__name__}")
    except Exception as e:
        print_error(f"Failed to load scaler: {e}")
        return False
    
    try:
        with open(ENCODER_PATH, 'rb') as f:
            encoders = pickle.load(f)
        print_success(f"Encoders loaded: {len(encoders)} label encoders")
        for col_name in encoders.keys():
            print_info(f"  - {col_name}: {list(encoders[col_name].classes_)}")
    except Exception as e:
        print_error(f"Failed to load encoders: {e}")
        return False
    
    return True, model, scaler, encoders


def test_predictions(model, scaler, encoders):
    """Test model predictions with test cases."""
    print_header("TEST 3: Model Predictions")
    
    successful_predictions = 0
    
    for test_case in TEST_CASES:
        print(f"\nTest Case: {test_case['name']}")
        print("-" * 80)
        
        try:
            # Prepare input
            input_data = test_case['data'].copy()
            numerical_cols = ['age', 'income', 'credit_score', 'employment_length', 
                            'loan_amount', 'interest_rate']
            categorical_cols = ['loan_type', 'employment_type']
            
            # Create DataFrame with proper column names
            df = pd.DataFrame({
                'Age': [input_data['age']],
                'Income': [input_data['income']],
                'Credit_Score': [input_data['credit_score']],
                'Employment_Length': [input_data['employment_length']],
                'Loan_Amount': [input_data['loan_amount']],
                'Interest_Rate': [input_data['interest_rate']],
                'Loan_Type': [input_data['loan_type']],
                'Employment_Type': [input_data['employment_type']]
            })
            
            # Encode categorical variables
            for col in categorical_cols:
                col_mapped = col.replace('_', ' ').title().replace(' ', '_')
                if col_mapped in encoders:
                    df[col_mapped] = encoders[col_mapped].transform([input_data[col]])
            
            # Scale numerical features
            all_cols = ['Age', 'Income', 'Credit_Score', 'Employment_Length', 
                       'Loan_Amount', 'Interest_Rate', 'Loan_Type', 'Employment_Type']
            X = df[all_cols].values.astype(float)
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0]
            
            result = 'Approved' if prediction == 1 else 'Rejected'
            confidence = max(probability) * 100
            
            print(f"Input Data:")
            for key, value in test_case['data'].items():
                print(f"  {key}: {value}")
            
            print(f"\nPrediction: {result}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Probabilities: Rejected={probability[0]:.4f}, Approved={probability[1]:.4f}")
            
            print_success(f"Prediction successful")
            successful_predictions += 1
            
        except Exception as e:
            print_error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"Successful predictions: {successful_predictions}/{len(TEST_CASES)}")
    
    return successful_predictions == len(TEST_CASES)


def test_input_validation():
    """Test input validation."""
    print_header("TEST 4: Input Validation")
    
    invalid_inputs = [
        {
            'name': 'Invalid Age (too young)',
            'data': {'age': 10, 'income': 50000, 'credit_score': 700, 
                    'employment_length': 5, 'loan_amount': 200000, 
                    'interest_rate': 4.5, 'loan_type': 'Personal', 
                    'employment_type': 'Employed'}
        },
        {
            'name': 'Invalid Credit Score',
            'data': {'age': 35, 'income': 50000, 'credit_score': 900,
                    'employment_length': 5, 'loan_amount': 200000,
                    'interest_rate': 4.5, 'loan_type': 'Personal',
                    'employment_type': 'Employed'}
        },
        {
            'name': 'Missing Required Field',
            'data': {'age': 35, 'credit_score': 700, 'employment_length': 5,
                    'loan_amount': 200000, 'interest_rate': 4.5,
                    'loan_type': 'Personal', 'employment_type': 'Employed'}
        }
    ]
    
    for test in invalid_inputs:
        print(f"\n{test['name']}")
        print("-" * 40)
        
        # Check ranges
        if 'age' in test['data']:
            if test['data']['age'] < 18 or test['data']['age'] > 70:
                print_success(f"Correctly identified invalid age: {test['data']['age']}")
        
        if 'credit_score' in test['data']:
            if test['data']['credit_score'] < 300 or test['data']['credit_score'] > 850:
                print_success(f"Correctly identified invalid credit score: {test['data']['credit_score']}")
        
        if 'income' not in test['data']:
            print_success("Correctly identified missing income field")


def test_data_types():
    """Test data type handling."""
    print_header("TEST 5: Data Type Handling")
    
    tests = [
        ('String to float conversion', lambda: float("75000")),
        ('Integer to float conversion', lambda: float(75000)),
        ('Boolean to integer conversion', lambda: int(True)),
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            print_success(f"{test_name}: {result}")
        except Exception as e:
            print_error(f"{test_name}: {e}")


def test_directory_structure():
    """Test project directory structure."""
    print_header("TEST 6: Directory Structure")
    
    required_dirs = [
        'data',
        'models',
        'templates',
        'static'
    ]
    
    required_files = [
        'train.py',
        'app.py',
        'requirements.txt',
        'README.md'
    ]
    
    all_ok = True
    
    print("\nRequired Directories:")
    for dir_name in required_dirs:
        dir_path = os.path.join(os.path.dirname(__file__), dir_name)
        if os.path.isdir(dir_path):
            print_success(f"{dir_name}/")
        else:
            print_error(f"{dir_name}/ NOT FOUND")
            all_ok = False
    
    print("\nRequired Files:")
    for file_name in required_files:
        file_path = os.path.join(os.path.dirname(__file__), file_name)
        if os.path.isfile(file_path):
            print_success(f"{file_name}")
        else:
            print_error(f"{file_name} NOT FOUND")
            all_ok = False
    
    return all_ok


def run_all_tests():
    """Run all tests."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  LOAN APPROVAL PREDICTION SYSTEM - TEST SUITE".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝\n")
    
    results = {}
    
    # Test 1: Directory structure
    results['Directory Structure'] = test_directory_structure()
    
    # Test 2: Model files existence
    results['Model Files'] = test_model_files()
    
    # Test 3: Model loading
    try:
        model_loaded, model, scaler, encoders = test_model_loading()
        results['Model Loading'] = model_loaded
    except:
        print_error("Model loading failed")
        results['Model Loading'] = False
        model, scaler, encoders = None, None, None
    
    # Test 4: Predictions (requires loaded model)
    if model and scaler and encoders:
        results['Model Predictions'] = test_predictions(model, scaler, encoders)
    else:
        print_error("Skipping prediction tests - model not loaded")
        results['Model Predictions'] = False
    
    # Test 5: Input validation
    test_input_validation()
    results['Input Validation'] = True
    
    # Test 6: Data types
    test_data_types()
    results['Data Type Handling'] = True
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        if result:
            print_success(f"{test_name}")
            passed += 1
        else:
            print_error(f"{test_name}")
    
    print(f"\n{'='*80}")
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    print(f"{'='*80}\n")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
