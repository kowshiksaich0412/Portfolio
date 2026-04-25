"""
House Price Prediction - Machine Learning Pipeline
Complete ML workflow with data preprocessing, EDA, model training, and evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD DATASET
# ============================================================================
def load_data(filepath):
    """
    Load the dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print("Loading dataset...")
    data = pd.read_csv(filepath)
    print(f"Dataset shape: {data.shape}")
    print(f"\nFirst few rows:\n{data.head()}")
    print(f"\nDataset info:\n{data.info()}")
    print(f"\nMissing values:\n{data.isnull().sum()}")
    return data


# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
def preprocess_data(data):
    """
    Preprocess the dataset:
    - Handle missing values
    - Encode categorical variables
    - Feature scaling
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X_processed, y, label_encoders dict, scaler object)
    """
    print("\n" + "="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Create a copy to avoid modifying original
    df = data.copy()
    
    # Separate features and target
    X = df.drop(['Id', 'Price'], axis=1)
    y = df['Price']
    
    # -------- Handle Missing Values --------
    print("\nHandling missing values...")
    missing_count = X.isnull().sum().sum()
    if missing_count == 0:
        print("No missing values found!")
    else:
        # Fill numerical columns with mean
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())
        
        # Fill categorical columns with mode
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        print(f"Missing values handled: {missing_count} replaced")
    
    # -------- Encode Categorical Variables --------
    print("\nEncoding categorical variables...")
    label_encoders = {}
    categorical_columns = X.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # -------- Feature Scaling --------
    print("\nApplying feature scaling...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Preprocessed features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    
    return X_scaled, y, label_encoders, scaler, X.columns


# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
def perform_eda(data, X_scaled, y, project_dir):
    """
    Perform Exploratory Data Analysis:
    - Display statistical summary
    - Create correlation heatmap
    - Plot feature distributions
    - Plot target distribution
    
    Args:
        data (pd.DataFrame): Original dataset
        X_scaled (pd.DataFrame): Scaled features
        y (pd.Series): Target variable
        project_dir (str): Project directory path
    """
    print("\n" + "="*70)
    print("EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)
    
    # Statistical Summary
    print("\nStatistical Summary of Original Dataset:")
    print(data.describe())
    
    # Prepare dataframe for correlation analysis (use original numeric data)
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Correlation Heatmap
    print("\nGenerating correlation heatmap...")
    corr_matrix = numeric_data.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                ax=axes[0, 0], square=True, cbar_kws={'label': 'Correlation'})
    axes[0, 0].set_title('Correlation Heatmap', fontweight='bold')
    
    # 2. Target Distribution
    print("Generating target distribution plot...")
    axes[0, 1].hist(y, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Price', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Target Variable Distribution (Price)', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. Area vs Price (Scatter plot)
    print("Generating Area vs Price scatter plot...")
    area_col = 'Area' if 'Area' in data.columns else data.columns[1]
    axes[1, 0].scatter(data[area_col], y, alpha=0.5, s=50, color='green')
    axes[1, 0].set_xlabel(area_col, fontweight='bold')
    axes[1, 0].set_ylabel('Price', fontweight='bold')
    axes[1, 0].set_title(f'{area_col} vs Price', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. Feature Statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Dataset Statistics:
    
    • Total samples: {len(data)}
    • Number of features: {len(X_scaled.columns)}
    • Price range: ${y.min():,.0f} - ${y.max():,.0f}
    • Mean price: ${y.mean():,.0f}
    • Median price: ${y.median():,.0f}
    
    Features:
    {chr(10).join([f'  • {col}' for col in X_scaled.columns])}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    eda_path = os.path.join(project_dir, 'eda_analysis.png')
    plt.savefig(eda_path, dpi=300, bbox_inches='tight')
    print(f"\nEDA visualization saved: {eda_path}")
    plt.close()


# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ============================================================================
# 5. MODEL TRAINING
# ============================================================================
def train_models(X_train, y_train):
    """
    Train multiple regression models.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        
    Returns:
        dict: Dictionary of trained models
    """
    print("\n" + "="*70)
    print("MODEL TRAINING")
    print("="*70)
    
    models = {}
    
    # 1. Linear Regression
    print("\n1. Training Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    models['Linear Regression'] = lr_model
    print("   ✓ Linear Regression trained")
    
    # 2. Decision Tree Regressor
    print("2. Training Decision Tree Regressor...")
    dt_model = DecisionTreeRegressor(random_state=42, max_depth=20)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    print("   ✓ Decision Tree Regressor trained")
    
    # 3. Random Forest Regressor
    print("3. Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, 
                                     max_depth=20, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    print("   ✓ Random Forest Regressor trained")
    
    return models


# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
def evaluate_models(models, X_test, y_test, feature_names):
    """
    Evaluate all trained models and display results.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (pd.DataFrame): Testing features
        y_test (pd.Series): Testing target
        feature_names (list): Names of features
        
    Returns:
        dict: Dictionary of model results
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    results = {}
    predictions = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        print("-" * 50)
        
        # Make predictions
        y_pred = model.predict(X_test)
        predictions[model_name] = y_pred
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[model_name] = {'R2': r2, 'RMSE': rmse}
        
        print(f"  R² Score: {r2:.4f}")
        print(f"  RMSE: ${rmse:,.2f}")
    
    # Display comparison table
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    results_df = pd.DataFrame(results).T
    print("\n" + results_df.to_string())
    
    return results, predictions


# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
def display_feature_importance(models, feature_names, project_dir):
    """
    Display feature importance for tree-based models.
    
    Args:
        models (dict): Dictionary of trained models
        feature_names (list): Names of features
        project_dir (str): Project directory path
    """
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Feature Importance Analysis', fontsize=14, fontweight='bold')
    
    # Decision Tree Feature Importance
    if 'Decision Tree' in models:
        print("\nDecision Tree Feature Importance:")
        dt_importance = models['Decision Tree'].feature_importances_
        dt_features = sorted(zip(feature_names, dt_importance), 
                            key=lambda x: x[1], reverse=True)
        
        features, importances = zip(*dt_features)
        axes[0].barh(features, importances, color='steelblue')
        axes[0].set_xlabel('Importance', fontweight='bold')
        axes[0].set_title('Decision Tree Feature Importance', fontweight='bold')
        axes[0].invert_yaxis()
        
        for feature, importance in dt_features:
            print(f"  {feature}: {importance:.4f}")
    
    # Random Forest Feature Importance
    if 'Random Forest' in models:
        print("\nRandom Forest Feature Importance:")
        rf_importance = models['Random Forest'].feature_importances_
        rf_features = sorted(zip(feature_names, rf_importance), 
                            key=lambda x: x[1], reverse=True)
        
        features, importances = zip(*rf_features)
        axes[1].barh(features, importances, color='forestgreen')
        axes[1].set_xlabel('Importance', fontweight='bold')
        axes[1].set_title('Random Forest Feature Importance', fontweight='bold')
        axes[1].invert_yaxis()
        
        for feature, importance in rf_features:
            print(f"  {feature}: {importance:.4f}")
    
    plt.tight_layout()
    fi_path = os.path.join(project_dir, 'feature_importance.png')
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    print(f"\nFeature importance visualization saved: {fi_path}")
    plt.close()


# ============================================================================
# 8. SAVE BEST MODEL
# ============================================================================
def save_best_model(models, results, feature_names, scaler, label_encoders, project_dir):
    """
    Save the best performing model and necessary preprocessing objects.
    
    Args:
        models (dict): Dictionary of trained models
        results (dict): Dictionary of model results
        feature_names (list): Names of features
        scaler (StandardScaler): Fitted scaler object
        label_encoders (dict): Dictionary of label encoders
        project_dir (str): Project directory path
    """
    print("\n" + "="*70)
    print("SAVING BEST MODEL")
    print("="*70)
    
    # Find best model based on R² score
    best_model_name = max(results, key=lambda x: results[x]['R2'])
    best_model = models[best_model_name]
    best_r2 = results[best_model_name]['R2']
    
    print(f"\nBest Model: {best_model_name}")
    print(f"R² Score: {best_r2:.4f}")
    
    # Create models directory if not exists
    models_dir = os.path.join(project_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Save best model
    model_path = os.path.join(models_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Best model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save label encoders
    encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✓ Label encoders saved: {encoders_path}")
    
    # Save feature names
    features_path = os.path.join(models_dir, 'feature_names.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(list(feature_names), f)
    print(f"✓ Feature names saved: {features_path}")
    
    # Save all models for comparison
    all_models_path = os.path.join(models_dir, 'all_models.pkl')
    with open(all_models_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"✓ All models saved: {all_models_path}")
    
    print("\nAll artifacts saved successfully!")


# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================
def main():
    """
    Main execution function - runs the complete ML pipeline.
    """
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - MACHINE LEARNING PROJECT")
    print("="*70)
    
    # Get the current directory and construct file path
    project_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(project_dir, 'House Price Prediction Dataset.csv')
    
    # Create models directory
    models_dir = os.path.join(project_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    
    # Load data
    data = load_data(csv_file)
    
    # Preprocess data
    X_scaled, y, label_encoders, scaler, feature_names = preprocess_data(data)
    
    # Perform EDA
    perform_eda(data, X_scaled, y, project_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results, predictions = evaluate_models(models, X_test, y_test, feature_names)
    
    # Display feature importance
    display_feature_importance(models, feature_names, project_dir)
    
    # Save best model
    save_best_model(models, results, feature_names, scaler, label_encoders, project_dir)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 'python app.py' to start the Flask web application")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Enter house features to get price predictions")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
