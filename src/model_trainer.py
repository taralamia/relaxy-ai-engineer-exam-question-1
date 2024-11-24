import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, matthews_corrcoef, confusion_matrix)
import joblib
import os
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                            ExtraTreesClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

class ModelTrainer:
    def __init__(self):
        self.transformed_data_dir = "artifacts/transformed_data"
        self.model_dir = "artifacts/models"
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000), # Increased max_iter
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "MLP": MLPClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "Extra Trees": ExtraTreesClassifier()
        }
    
    def create_directories(self):
        """Create necessary directories for storing models"""
        os.makedirs(self.model_dir, exist_ok=True)
    
    def load_transformed_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from the transformed data directory"""
        try:
            X = pd.read_csv(os.path.join(self.transformed_data_dir, "X_transformed.csv"))
            y = pd.read_csv(os.path.join(self.transformed_data_dir, "y_transformed.csv")).iloc[:, 0]
            return X, y
        except Exception as e:
            print(f"Error loading transformed data: {str(e)}")
            raise e
    
    def evaluate_model(self, model: Any, X_train: pd.DataFrame,
                       y_train: pd.Series, X_test: pd.DataFrame, 
                       y_test: pd.Series) -> Dict[str, float]:
        """Train and evaluate a single model"""
        # Clean the data by replacing infinities and dropping NaN values
        X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
        X_test = X_test.replace([np.inf, -np.inf], np.nan).dropna()
        y_train = y_train.dropna()
        y_test = y_test.dropna()

        # Ensure dimensions match after cleaning
        X_train, y_train = X_train.align(y_train, axis=0, join='inner')
        X_test, y_test = X_test.align(y_test, axis=0, join='inner')

        # Convert data to numpy array if using CatBoost
        if isinstance(model, CatBoostClassifier):
            X_train = X_train.to_numpy() 
            X_test = X_test.to_numpy()

        # Fit the model and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'Test Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'MCC': matthews_corrcoef(y_test, y_pred),
            'Confusion Matrix': confusion_matrix(y_test, y_pred).ravel()
        }
        return metrics
    
    def save_models(self, results: Dict[str, Dict]):
        """Save all models and their metrics"""
        try: 
            # Save all models
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{model_name.lower().replace(' ', '_')}.joblib")
                joblib.dump(model, model_path)
            
            # Save metrics
            results_df = pd.DataFrame(results).T
            results_df.to_csv(os.path.join(self.model_dir, "model_metrics.csv"))
            
            print(f"Models and metrics saved to {self.model_dir}")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise e
    
    def initiate_model_training(self) -> Dict[str, Dict]:
        """Train and evaluate all models"""
        try:
            # Create directories
            self.create_directories()
            
            # Load transformed data
            X, y = self.load_transformed_data()
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            results = {}
            for model_name, model in self.models.items():
                print(f"Training and evaluating {model_name}...")
                metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)
                results[model_name] = metrics
            
            # Save all models and their metrics
            self.save_models(results)
            
            print("Model training completed successfully")
            return results
            
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise e

if __name__ == "__main__":
    model_trainer = ModelTrainer()
    results = model_trainer.initiate_model_training()
    
    # Print results
    results_df = pd.DataFrame(results).T
    print("\nModel Evaluation Results:")
    print(results_df)
