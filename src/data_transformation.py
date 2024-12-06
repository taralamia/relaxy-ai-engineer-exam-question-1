import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
import joblib
from typing import Tuple

class DataTransformation:
    def __init__(self):
        self.ingested_data_dir = "artifacts/ingested_data"
        self.transformed_data_dir = "artifacts/transformed_data"
        self.numerical_features = [
            'income_annum', 'loan_amount', 'loan_term', 
            'cibil_score', 'residential_assets_value', 
            'commercial_assets_value', 'luxury_assets_value', 
            'bank_asset_value', 'income_per_dependent'
        ]
        self.categorical_features = ['education', 'self_employed', 'cibil_category']
    
    def create_directories(self):
        """Create necessary directories for storing transformed data."""
        os.makedirs(self.transformed_data_dir, exist_ok=True)
    
    def load_ingested_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load data from the ingested data directory."""
        try:
            X = pd.read_csv(os.path.join(self.ingested_data_dir, "X.csv"))
            y = pd.read_csv(os.path.join(self.ingested_data_dir, "y.csv")).iloc[:, 0]
            return X, y
        except Exception as e:
            print(f"Error loading ingested data: {str(e)}")
            raise e
    
    def save_transformed_data(self, X_transformed: pd.DataFrame, y: pd.Series, preprocessor: ColumnTransformer):
        """Save the transformed data and preprocessing objects."""
        try:
            pd.DataFrame(X_transformed).to_csv(os.path.join(self.transformed_data_dir, "X_transformed.csv"), index=False)
            y.to_csv(os.path.join(self.transformed_data_dir, "y_transformed.csv"), index=False)
            joblib.dump(preprocessor, os.path.join(self.transformed_data_dir, "preprocessor.joblib"))
            print(f"Transformed data and preprocessor saved to {self.transformed_data_dir}")
        except Exception as e:
            print(f"Error saving transformed data: {str(e)}")
            raise e
    
    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Perform feature engineering on the dataset."""
        X = X.copy()
        # Derived features
        X['loan_to_income_ratio'] = X['loan_amount'] / X['income_annum']
        X['total_asset_value'] = (
            X['residential_assets_value'] +
            X['commercial_assets_value'] +
            X['luxury_assets_value'] +
            X['bank_asset_value']
        )
        X['income_per_dependent'] = X['income_annum'] / (X['no_of_dependents'] + 1)
        
        # CIBIL category creation
        def cibil_category(score):
            if score < 500:
                return 'Low'
            elif 500 <= score <= 700:
                return 'Medium'
            return 'High'
        
        X['cibil_category'] = X['cibil_score'].apply(cibil_category)
        return X
    
    def get_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline for numerical and categorical features."""
        return ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )
    
    def initiate_data_transformation(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load, transform, and save data."""
        self.create_directories()
        X, y = self.load_ingested_data()
        X = self.feature_engineering(X)
        preprocessor = self.get_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        self.save_transformed_data(X_transformed, y, preprocessor)
        return X_transformed, y

if __name__ == "__main__":
    data_transformation = DataTransformation()
    X_transformed, y = data_transformation.initiate_data_transformation()
    print(f"Transformed features shape: {X_transformed.shape}")
