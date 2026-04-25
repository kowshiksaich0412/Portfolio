# ============================================================================
# Loan Approval Prediction - Configuration
# ============================================================================
# Customize this file to adjust model behavior and Flask settings

# ============================================================================
# FLASK CONFIGURATION
# ============================================================================

# Server settings
FLASK_ENV = 'development'
FLASK_DEBUG = True
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000

# Application settings
SECRET_KEY = 'loan-approval-secret-key-2026'
MAX_CONTENT_LENGTH = 16777216  # 16MB max request size

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Logistic Regression
LOGISTIC_REGRESSION_MAX_ITER = 1000
LOGISTIC_REGRESSION_RANDOM_STATE = 42

# Decision Tree
DECISION_TREE_MAX_DEPTH = 10
DECISION_TREE_RANDOM_STATE = 42

# Random Forest
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 15
RANDOM_FOREST_RANDOM_STATE = 42

# ============================================================================
# DATA PROCESSING
# ============================================================================

# Train-test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
STRATIFY_TARGET = True

# Feature scaling method
SCALER_TYPE = 'StandardScaler'  # Options: 'StandardScaler', 'MinMaxScaler'

# ============================================================================
# VISUALIZATION
# ============================================================================

# Plot settings
PLOT_DPI = 300
PLOT_FORMAT = 'png'
FIGURE_SIZE = (12, 8)

# Colors
COLOR_APPROVED = '#4ECDC4'
COLOR_REJECTED = '#FF6B6B'
COLOR_PRIMARY = '#2563eb'
COLOR_SECONDARY = '#1e40af'

# ============================================================================
# PATHS
# ============================================================================

DATA_PATH = 'data/loan_data.csv'
MODEL_PATH = 'models/best_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/label_encoders.pkl'
PLOTS_PATH = 'static/plots'

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'  # Options: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_FILE = 'app.log'

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Input validation ranges
MIN_AGE = 18
MAX_AGE = 70
MIN_INCOME = 10000
MAX_INCOME = 500000
MIN_CREDIT_SCORE = 300
MAX_CREDIT_SCORE = 850
MIN_EMPLOYMENT_LENGTH = 0
MAX_EMPLOYMENT_LENGTH = 50
MIN_LOAN_AMOUNT = 5000
MAX_LOAN_AMOUNT = 1000000
MIN_INTEREST_RATE = 2.0
MAX_INTEREST_RATE = 10.0

# Valid categorical values
VALID_LOAN_TYPES = ['Personal', 'Auto', 'Business']
VALID_EMPLOYMENT_TYPES = ['Employed', 'Self-Employed']

# ============================================================================
# NOTES
# ============================================================================
#
# To use these settings in your code:
#
#   import os
#   from dotenv import load_dotenv
#
#   load_dotenv()
#   FLASK_PORT = int(os.getenv('FLASK_PORT', 5000))
#   SECRET_KEY = os.getenv('SECRET_KEY')
#
# Or copy to .env file and use python-dotenv
