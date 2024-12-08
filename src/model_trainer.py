import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, confusion_matrix)
import joblib
import mlflow
from mlflow.tracking import MlflowClient
import os
import logging
from typing import Dict, Any, Tuple
import datetime
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
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.transformed_data_dir = "artifacts/transformed_data"
        self.model_dir = "artifacts/models"
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(),
            "LightGBM": LGBMClassifier(),
            "CatBoost": CatBoostClassifier(verbose=0),
            "SVC": SVC(probability=True, kernel='rbf', C=1.0),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "Naive Bayes": GaussianNB(),
            "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500),
            "AdaBoost": AdaBoostClassifier(),
            "Extra Trees": ExtraTreesClassifier()
        }
        self.best_model = None
        self.best_model_name = None
        self.best_f1_score = 0

    def create_directories(self):
        """Create necessary directories for storing models."""
        os.makedirs(self.model_dir, exist_ok=True)

    def load_transformed_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from the transformed data directory and log data characteristics."""
        try:
            X = pd.read_csv(os.path.join(self.transformed_data_dir, "X_transformed.csv"))
            y = pd.read_csv(os.path.join(self.transformed_data_dir, "y_transformed.csv")).iloc[:, 0]

            # Log data characteristics
            feature_distributions = X.describe().to_dict()
            missing_values = X.isnull().sum().to_dict()
            mlflow.log_dict(feature_distributions, "feature_distributions.json")
            mlflow.log_dict(missing_values, "missing_values.json")

            return X, y
        except Exception as e:
            print(f"Error loading transformed data: {str(e)}")
            raise e

    def evaluate_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                       X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train and evaluate a single model."""
        # Clean the data
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

        # Fit and predict
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).ravel().tolist()
        }

        # Track the best model based on F1 score
        if metrics['f1_score'] > self.best_f1_score:
            self.best_f1_score = metrics['f1_score']
            self.best_model = model
            self.best_model_name = model.__class__.__name__

        return metrics

    def save_models(self, results: Dict[str, Dict]):
        """Save all models and their metrics."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.model_dir, exist_ok=True)

            # Save all models
            for model_name, model in self.models.items():
                model_path = os.path.join(self.model_dir, f"{model_name.lower().replace(' ', '_')}.joblib")
                joblib.dump(model, model_path)

            # Save metrics
            results_df = pd.DataFrame(results).T
            results_df.to_csv(os.path.join(self.model_dir, f"model_metrics_{timestamp}.csv"))

            print(f"Models and metrics saved to {self.model_dir}")

        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise e

    def register_best_model(self):
        """Register the best model in MLFlow Model Registry."""
        if self.best_model is None:
            print("No best model to register.")
            return

        client = MlflowClient()
        model_name = "loan_approval_model"

        # Log and register the best model
        mlflow.sklearn.log_model(
            self.best_model,
            artifact_path="model",
            registered_model_name=model_name
        )

        # Transition to production
        try:
            latest_versions = client.get_latest_versions(model_name, stages=["None"])

            # Check if there are any versions of the model
            if latest_versions:
                latest_version = latest_versions[0]  # Get the latest version
                self.logger.info(f"Latest version: {latest_version.version}")
                
                # Transition the model to production
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Production"
                )
                self.logger.info(f"Model version {latest_version.version} transitioned to Production.")

            else:
                self.logger.warning("No models found in the 'None' stage.")

        except Exception as e:
            self.logger.error(f"Error retrieving model versions: {str(e)}")

    def initiate_model_training(self) -> Dict[str, Dict]:
        """Train and evaluate all models while logging experiments in MLFlow."""
        try:
            self.create_directories()

            X, y = self.load_transformed_data()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            results = {}
            for model_name, model in self.models.items():
                print(f"Training and evaluating {model_name}...")

                with mlflow.start_run(nested=True):
                    metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)
                    results[model_name] = metrics

                    # Log metrics
                    for metric_name, value in metrics.items():
                        if isinstance(value, (float, int)):
                            try:
                                mlflow.log_metric(metric_name, value)
                            except Exception as e:
                                print(f"Error logging metric {metric_name}: {e}")    
                        elif isinstance(value, list):
                            try:
                                mlflow.log_dict({"values": value}, f"{metric_name}.json")
                            except Exception as e:
                                  print(f"Error logging list metric {metric_name}: {e}")
                    

                    # Log model
                    try:
                         mlflow.sklearn.log_model(model, artifact_path=f"models/{model_name.lower().replace(' ', '_')}")
                    except Exception as e:
                         print(f"Error logging model {model_name}: {e}")

                    # Log parameters
                    if hasattr(model, "get_params"):
                        mlflow.log_params(model.get_params())

            self.save_models(results)
            self.register_best_model()
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
