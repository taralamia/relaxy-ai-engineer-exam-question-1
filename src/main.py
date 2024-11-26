import mlflow
import mlflow.sklearn
from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
import pandas as pd
import os
import logging
from datetime import datetime

def setup_logging():
    """Setup logging configuration"""
    log_filename = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'artifacts',
        'artifacts/ingested_data',
        'artifacts/transformed_data',
        'artifacts/models',
        'logs'
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")

def save_pipeline_summary(ingestion_summary: dict, transformation_summary: dict, 
                          training_summary: pd.DataFrame):
    """Save pipeline execution summary"""
    summary = {
        'data_ingestion': ingestion_summary,
        'data_transformation': transformation_summary,
        'model_training': training_summary.to_dict()
    }
    
    # Use nested DataFrame conversion for compatibility
    summary_df = pd.DataFrame({
        "Stage": ["Data Ingestion", "Data Transformation", "Model Training"],
        "Details": [ingestion_summary, transformation_summary, training_summary.to_dict()],
    })
    summary_path = f"artifacts/pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Pipeline summary saved to {summary_path}")

def main():
    try:
        # Setup logging
        logger = setup_logging()
        logger.info("Starting ML pipeline...")

        # Set MLFlow tracking URI for cloud storage (S3)
        mlflow.set_tracking_uri("http://18.141.222.211:5000")  # MLFlow server URL
        mlflow.set_experiment("Loan_Approval_Experiment")
        mlflow.start_run()

        # Create directories
        create_directories()

        # Initialize pipeline components
        data_ingestion = DataIngestion('/root/.local/relaxy-ai-engineer-exam-question-1/src/dataset/loan_approval_dataset.csv')
        data_transformation = DataTransformation()
        model_trainer = ModelTrainer()

        # 1. Data Ingestion
        logger.info("Step 1: Data Ingestion")
        X, y = data_ingestion.initiate_data_ingestion()
        ingestion_summary = {
            'features_shape': X.shape,
            'target_shape': y.shape,
            'ingested_data_path': data_ingestion.ingested_data_dir
        }
        logger.info(f"Data ingestion completed. Data saved to {data_ingestion.ingested_data_dir}")

        # 2. Data Transformation
        logger.info("Step 2: Data Transformation")
        X_transformed, y = data_transformation.initiate_data_transformation()
        transformation_summary = {
            'transformed_features_shape': X_transformed.shape,
            'transformed_data_path': data_transformation.transformed_data_dir
        }
        logger.info(f"Data transformation completed. Data saved to {data_transformation.transformed_data_dir}")

        # 3. Model Training and Evaluation
        logger.info("Step 3: Model Training and Evaluation")
        results = model_trainer.initiate_model_training()
        results_df = pd.DataFrame(results).T
        print("Results DataFrame:", results_df)

        # Log metrics and parameters with MLFlow
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            mlflow.log_metric(metric, results_df.loc[0, metric])

        mlflow.log_params({
            'training_data_shape': X.shape,
            'transformed_data_shape': X_transformed.shape
        })

        # Log the model as an artifact
        mlflow.sklearn.log_model(model_trainer.best_model, "model")

        # Save results
        results_path = os.path.join(model_trainer.model_dir, 'model_evaluation_results.csv')
        results_df.to_csv(results_path)
        logger.info(f"Model evaluation results saved to {results_path}")

        # Convert 'F1 Score' to numeric and drop invalid rows
        results_df['F1 Score'] = pd.to_numeric(results_df['F1 Score'], errors='coerce')
        results_df = results_df.dropna(subset=['F1 Score'])

        # Find the best model
        best_model = results_df['F1 Score'].idxmax()
        logger.info(f"Best performing model: {best_model} (F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f})")

        # Save pipeline summary
        save_pipeline_summary(ingestion_summary, transformation_summary, results_df)

        logger.info("ML pipeline completed successfully!")

        # Print final summary
        print("\nPipeline Summary:")
        print("-" * 50)
        print(f"Data Ingestion:")
        print(f"- Features shape: {X.shape}")
        print(f"- Target shape: {y.shape}")
        print(f"\nData Transformation:")
        print(f"- Transformed features shape: {X_transformed.shape}")
        print(f"\nModel Training:")
        print(f"- Best model: {best_model}")
        print(f"- Best F1 Score: {results_df.loc[best_model, 'F1 Score']:.4f}")
        print(f"\nArtifacts saved in:")
        print(f"- Ingested data: {data_ingestion.ingested_data_dir}")
        print(f"- Transformed data: {data_transformation.transformed_data_dir}")
        print(f"- Models: {model_trainer.model_dir}")
        print("-" * 50)

        # End MLFlow run
        mlflow.end_run()

    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
