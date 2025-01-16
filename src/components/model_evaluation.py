from dataclasses import dataclass
from pathlib import Path
import os,json
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from src.constants import *
from src.utils.common import *
import dagshub
dagshub.init(repo_owner='Syam916', repo_name='mlops', mlflow=True)


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    test_data_transformed_path: Path
    test_target_path: Path
    best_model_path: Path
    evaluation_results_path: Path
    mlflow_uri: str

class ConfigurationManager:
    def __init__(self, config=CONFIG_FILE_PATH):
        self.config = read_yml_file(CONFIG_FILE_PATH)
        create_directories([self.config.artifacts_directory])

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_transformed_path=config.test_data_transformed_path,
            test_target_path=config.test_target_path,
            best_model_path=config.best_model_path,
            evaluation_results_path=config.evaluation_results_path,
            mlflow_uri=config.mlflow_uri
        )

        return model_evaluation_config

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_model(self):
        """
        Evaluate the best model and log metrics to MLflow
        """
        try:
            # Load test data and best model
            X_test = np.load(self.config.test_data_transformed_path)
            y_test = np.load(self.config.test_target_path)

            with open(self.config.best_model_path, 'rb') as f:
                model = pickle.load(f)

            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment("house_price_prediction")

            with mlflow.start_run():
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Log metrics to MLflow
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

                # Log model parameters
                model_params = model.get_params()
                mlflow.log_params(model_params)

                # Log model name and type
                mlflow.log_param("model_type", type(model).__name__)

                # Log the model itself
                mlflow.sklearn.log_model(model, "best_model")

                logger.info("Model evaluation completed and metrics logged to MLflow")

                # Save evaluation results locally
                results = {
                    'R2_Score': r2,
                    'RMSE': rmse,
                    'MAE': mae,
                    'Model_Type': type(model).__name__,
                    'Model_Parameters': model_params
                }

                with open(self.config.evaluation_results_path, 'w') as f:
                    json.dump(results, f, indent=4)

                logger.info("Evaluation results saved locally")
                
                return results

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise e

# if __name__ == "__main__":
#     try:
#         config = ConfigurationManager()
#         model_evaluation_config = config.get_model_evaluation_config()
#         model_evaluation = ModelEvaluation(config=model_evaluation_config)
        
#         evaluation_results = model_evaluation.evaluate_model()
        
#         print("\nModel Performance Metrics:")
#         print(f"R² Score: {evaluation_results['R2_Score']:.4f}")
#         print(f"RMSE: ${evaluation_results['RMSE']:,.2f}")
#         print(f"MAE: ${evaluation_results['MAE']:,.2f}")
#         print(f"\nModel Type: {evaluation_results['Model_Type']}")
        
#     except Exception as e:
#         raise e