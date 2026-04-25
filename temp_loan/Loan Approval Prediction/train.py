"""
Loan Approval Prediction Model Training Script
Handles data loading, preprocessing, EDA, model training, and evaluation.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'loan_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'label_encoders.pkl')
PLOTS_PATH = os.path.join(os.path.dirname(__file__), 'static', 'plots')

os.makedirs(PLOTS_PATH, exist_ok=True)


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_cols = []
        self.numerical_cols = []
        
    def load_data(self):
        """Load dataset from CSV file."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        print(f"\nFirst few rows:\n{self.df.head()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        return self.df
    
    def preprocess(self):
        """Handle missing values, encode categorical variables, normalize features."""
        print("\n" + "="*80)
        print("PREPROCESSING DATA")
        print("="*80)
        
        # Separate features and target
        X = self.df.drop('Approval_Status', axis=1)
        y = self.df['Approval_Status']
        
        # Identify categorical and numerical columns
        self.categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"\nCategorical columns: {self.categorical_cols}")
        print(f"Numerical columns: {self.numerical_cols}")
        
        # Handle missing values (forward fill for numerical, mode for categorical)
        for col in self.numerical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mean(), inplace=True)
        
        for col in self.categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].mode()[0], inplace=True)
        
        print("\nMissing values after handling:")
        print(X.isnull().sum())
        
        # Encode categorical variables
        print("\nEncoding categorical variables...")
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        # Normalize numerical features
        print("\nNormalizing numerical features...")
        X[self.numerical_cols] = self.scaler.fit_transform(X[self.numerical_cols])
        
        print(f"\nPreprocessed data shape: {X.shape}")
        print(f"Target distribution:\n{y.value_counts()}")
        
        return X, y


# ============================================================================
# Exploratory Data Analysis (EDA)
# ============================================================================
class EDA:
    """Performs exploratory data analysis."""
    
    def __init__(self, df, y, plots_path):
        self.df = df
        self.y = y
        self.plots_path = plots_path
        
    def class_distribution(self):
        """Analyze class distribution."""
        print("\n" + "="*80)
        print("EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        print("\nClass Distribution:")
        print(self.y.value_counts())
        print("\nClass Distribution (%):")
        print(self.y.value_counts(normalize=True) * 100)
        
        # Plot class distribution
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        self.y.value_counts().plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4'])
        ax.set_title('Loan Approval Status Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Approval Status (0=Rejected, 1=Approved)')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['Rejected', 'Approved'], rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        print("✓ Class distribution plot saved")
        
    def feature_relationships(self):
        """Analyze feature relationships with target variable."""
        print("\nAnalyzing feature relationships...")
        
        # Create correlation heatmap
        df_with_target = self.df.copy()
        df_with_target['Approval_Status'] = self.y
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(df_with_target.corr(), annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        print("✓ Correlation heatmap saved")
        
        # Feature distributions by approval status
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, col in enumerate(numerical_cols[:6]):
            df_with_target[df_with_target['Approval_Status'] == 0][col].hist(
                ax=axes[idx], alpha=0.6, label='Rejected', bins=20, color='#FF6B6B'
            )
            df_with_target[df_with_target['Approval_Status'] == 1][col].hist(
                ax=axes[idx], alpha=0.6, label='Approved', bins=20, color='#4ECDC4'
            )
            axes[idx].set_title(f'{col} Distribution by Approval Status')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_path, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        print("✓ Feature distributions plot saved")


# ============================================================================
# Model Training
# ============================================================================
class ModelTrainer:
    """Trains multiple ML models."""
    
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
        }
        self.trained_models = {}
        self.results = {}
        
    def train(self, X_train, y_train):
        """Train all models."""
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            print(f"✓ {model_name} trained successfully")
            
    def evaluate(self, X_test, y_test):
        """Evaluate all models."""
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        for model_name, model in self.trained_models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'conf_matrix': confusion_matrix(y_test, y_pred)
            }
            
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"ROC-AUC:   {roc_auc:.4f}")
            
            print(f"\nConfusion Matrix:\n{self.results[model_name]['conf_matrix']}")
            print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    def plot_confusion_matrices(self, plots_path):
        """Plot confusion matrices for all models."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (model_name, metrics) in enumerate(self.results.items()):
            cm = metrics['conf_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, annot_kws={'size': 12})
            axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices plot saved")
    
    def plot_roc_curves(self, X_test, y_test, plots_path):
        """Plot ROC curves for all models."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
        
        for idx, (model_name, model) in enumerate(self.trained_models.items()):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=colors[idx], lw=2,
                   label=f'{model_name} (AUC = {roc_auc:.4f})')
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, 'roc_curves.png'), dpi=300, bbox_inches='tight')
        print("✓ ROC curves plot saved")
    
    def get_best_model(self):
        """Return the best model based on F1-score."""
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        print(f"\n{'='*80}")
        print(f"Best Model: {best_model_name} (F1-Score: {self.results[best_model_name]['f1']:.4f})")
        print(f"{'='*80}")
        return best_model_name, self.trained_models[best_model_name]


# ============================================================================
# Model Saving
# ============================================================================
def save_model(model, model_path, scaler, scaler_path, encoders, encoder_path):
    """Save trained model and preprocessing objects."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_path}")
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")
    
    # Save encoders
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"✓ Label encoders saved to {encoder_path}")


# ============================================================================
# Summary Report
# ============================================================================
def print_summary_report(results):
    """Print a summary report of all models."""
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    summary_df = pd.DataFrame({
        'Model': results.keys(),
        'Accuracy': [v['accuracy'] for v in results.values()],
        'Precision': [v['precision'] for v in results.values()],
        'Recall': [v['recall'] for v in results.values()],
        'F1-Score': [v['f1'] for v in results.values()],
        'ROC-AUC': [v['roc_auc'] for v in results.values()]
    })
    
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*80)


# ============================================================================
# Main Pipeline
# ============================================================================
def main():
    """Execute the complete ML pipeline."""
    print("\n" + "="*80)
    print("LOAN APPROVAL PREDICTION - MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Step 1: Load and preprocess data
    processor = DataProcessor(DATA_PATH)
    processor.load_data()
    X, y = processor.preprocess()
    
    # Step 2: Exploratory Data Analysis
    eda = EDA(X, y, PLOTS_PATH)
    eda.class_distribution()
    eda.feature_relationships()
    
    # Step 3: Train-test split
    print("\n" + "="*80)
    print("TRAIN-TEST SPLIT")
    print("="*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    
    # Step 4: Train models
    trainer = ModelTrainer()
    trainer.train(X_train, y_train)
    
    # Step 5: Evaluate models
    trainer.evaluate(X_test, y_test)
    
    # Step 6: Plot results
    trainer.plot_confusion_matrices(PLOTS_PATH)
    trainer.plot_roc_curves(X_test, y_test, PLOTS_PATH)
    
    # Step 7: Get and save best model
    best_model_name, best_model = trainer.get_best_model()
    save_model(best_model, MODEL_PATH, processor.scaler, SCALER_PATH,
              processor.label_encoders, ENCODER_PATH)
    
    # Step 8: Print summary
    print_summary_report(trainer.results)
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"✓ All plots saved to {PLOTS_PATH}")
    print(f"✓ Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
