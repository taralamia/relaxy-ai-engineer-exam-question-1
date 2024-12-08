import mlflow
import os
import logging
from datetime import datetime
from data_ingestion import DataIngestion
from data_transformation import DataTransformation
from model_trainer import ModelTrainer
import pandas as pd


def setup_logging():
    """Setup logging configuration."""
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
    """Create necessary directories for the project."""
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
    """Save pipeline execution summary."""
    summary_path = f"artifacts/pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary = pd.DataFrame({
        "Stage": ["Data Ingestion", "Data Transformation", "Model Training"],
        "Details": [ingestion_summary, transformation_summary, training_summary.to_dict()]
    })
    summary.to_csv(summary_path, index=False)
    logging.info(f"Pipeline summary saved to {summary_path}")


def main():
    logger = setup_logging()
    logger.info("Starting ML pipeline...")

    try:
        # Set MLFlow tracking URI and experiment
        mlflow.set_tracking_uri("http://3.1.50.170:5000")  # Update with your MLFlow server URI
        mlflow.set_experiment("Loan_Approval_Experiment")

        # Start a new MLFlow run
        with mlflow.start_run() as run:
            # Create directories
            create_directories()

            # Initialize pipeline components
            data_ingestion = DataIngestion('/root/.local/relaxy-ai-engineer-exam-question-1/src/dataset/loan_approval_dataset.csv')
            data_transformation = DataTransformation()
            model_trainer = ModelTrainer()

            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            X, y = data_ingestion.initiate_data_ingestion()
            ingestion_summary = {
                'features_shape': X.shape,
                'target_shape': y.shape,
                'ingested_data_path': data_ingestion.ingested_data_dir
            }
            logger.info(f"Data ingestion completed. Data saved to {data_ingestion.ingested_data_dir}")

            # Log data characteristics for versioning
            logger.info("Logging data versioning details...")
            mlflow.log_dict(X.describe().to_dict(), "data_versioning/feature_distributions.json")
            mlflow.log_dict(X.isnull().sum().to_dict(), "data_versioning/missing_values.json")
            mlflow.log_param("data_ingestion_path", data_ingestion.ingested_data_dir)

            # Step 2: Data Transformation
            logger.info("Step 2: Data Transformation")
            X_transformed, y_transformed = data_transformation.initiate_data_transformation()
            transformation_summary = {
                'transformed_features_shape': X_transformed.shape,
                'transformed_data_path': data_transformation.transformed_data_dir
            }
            logger.info(f"Data transformation completed. Data saved to {data_transformation.transformed_data_dir}")

            # Step 3: Model Training and Evaluation
            logger.info("Step 3: Model Training and Evaluation")
            results = model_trainer.initiate_model_training()
            results_df = pd.DataFrame(results).T

            # Step 4: Register the Best Model
            logger.info("Registering the best model in MLFlow Model Registry...")
            model_trainer.register_best_model()

            # Step 5: Transition the Best Model to Production
            logger.info("Transitioning the best model to production stage...")
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            model_name = "loan_approval_model"
            latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
            client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Production"
            )
            logger.info(f"Best model '{model_name}' version {latest_version.version} transitioned to Production.")

            # Save pipeline summary
            save_pipeline_summary(ingestion_summary, transformation_summary, results_df)

            # Log final outputs
            logger.info("ML pipeline completed successfully!")
            print("\nPipeline Summary:")
            print("-" * 50)
            print(f"Data Ingestion:")
            print(f"- Features shape: {X.shape}")
            print(f"- Target shape: {y.shape}")
            print(f"\nData Transformation:")
            print(f"- Transformed features shape: {X_transformed.shape}")
            print(f"\nModel Training:")
            print(f"- Results saved in: {model_trainer.model_dir}")
            print("-" * 50)

    except Exception as e:
        logger.error(f"Error in ML pipeline: {str(e)}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
