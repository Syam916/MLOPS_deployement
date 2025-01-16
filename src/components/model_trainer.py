from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import pickle
from datetime import datetime
from src.constants import *
from src.utils.common import *

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    train_target_path: Path
    test_target_path: Path
    model_dir: Path
    model_params: dict
    cv_folds: int

class ConfigurationManager:
    def __init__(self, config=CONFIG_FILE_PATH):
        self.config = read_yml_file(CONFIG_FILE_PATH)
        create_directories([self.config.artifacts_directory])

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            train_target_path=config.train_target_path,
            test_target_path=config.test_target_path,
            model_dir=config.model_dir,
            model_params=config.model_params,
            cv_folds=config.cv_folds
        )

        return model_trainer_config

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def get_base_models(self):
        """
        Initialize base models for grid search
        """
        models = {
            'LinearRegression': LinearRegression(),
            'Lasso': Lasso(),
            'Ridge': Ridge(),
            'SVR': SVR()
        }
        return models

    def train_with_grid_search(self, X_train, y_train, model_name, base_model, param_grid):
        """
        Train model using grid search for hyperparameter tuning
        """
        logger.info(f"Starting grid search for {model_name}")
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.config.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        logger.info(f"Best score for {model_name}: {-grid_search.best_score_}")
        
        return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_

    def save_model_results(self, model_dir: str, model_results: dict):
        """
        Save model results in a formatted text file
        """
        results_file = os.path.join(model_dir, 'model_results.txt')
        with open(results_file, 'w') as f:
            f.write("MODEL TRAINING RESULTS\n")
            f.write("=====================\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Cross-validation folds: {self.config.cv_folds}\n\n")
            
            for model_name, result in model_results.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * (len(model_name) + 7) + "\n")
                f.write(f"Best Parameters: {result['best_params']}\n")
                f.write(f"MSE Score: {result['best_score']}\n")
                f.write(f"RMSE Score: {np.sqrt(result['best_score'])}\n\n")

    def find_best_model(self, model_results: dict):
        """
        Find the best performing model based on MSE
        """
        best_score = float('inf')
        best_model_name = None
        
        for model_name, result in model_results.items():
            current_score = result['best_score']
            if current_score < best_score:
                best_score = current_score
                best_model_name = model_name
        
        return best_model_name

    def train(self):
        try:
            # Load data
            X_train = np.load(self.config.train_data_path)
            X_test = np.load(self.config.test_data_path)
            y_train = np.load(self.config.train_target_path)
            y_test = np.load(self.config.test_target_path)
            
            logger.info("Loaded training and test data")

            # Create timestamped model directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            model_dir = self.config.model_dir
            os.makedirs(model_dir, exist_ok=True)
            
            # Get base models
            base_models = self.get_base_models()
            trained_models = {}
            model_results = {}
            
            # Train each model with grid search
            for name, model in base_models.items():
                logger.info(f"\nTraining {name}...")
                
                # Get parameter grid from config
                param_grid = self.config.model_params.get(name, {})
                
                if param_grid:  # If parameters are specified in config
                    # Perform grid search
                    best_model, best_params, best_score = self.train_with_grid_search(
                        X_train, y_train, name, model, param_grid
                    )
                    
                    # Save results
                    model_results[name] = {
                        'best_params': best_params,
                        'best_score': best_score,
                        'model': best_model
                    }
                    
                else:  # If no parameters specified, use default
                    logger.info(f"No parameter grid specified for {name}, using default parameters")
                    model.fit(X_train, y_train)
                    best_model = model
                    
                    model_results[name] = {
                        'best_params': 'default',
                        'best_score': float('inf'),
                        'model': best_model
                    }
                
                # Save individual model
                model_path = os.path.join(model_dir, f"{name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(best_model, f)
                
                trained_models[name] = model_path
                logger.info(f"Saved {name} model to {model_path}")
            
            # Save model results in text format
            self.save_model_results(model_dir, model_results)
            
            # Find and save best model
            best_model_name = self.find_best_model(model_results)
            best_model = model_results[best_model_name]['model']
            best_model_path = os.path.join(model_dir, 'best_model.pkl')
            
            with open(best_model_path, 'wb') as f:
                pickle.dump(best_model, f)
            
            logger.info(f"Best performing model: {best_model_name}")
            logger.info(f"Saved best model to: {best_model_path}")
            
            # Save model paths
            paths_file = os.path.join(model_dir, 'model_paths.pkl')
            with open(paths_file, 'wb') as f:
                pickle.dump({
                    'individual_models': trained_models,
                    'best_model': best_model_path
                }, f)
            
            logger.info(f"Model training completed. All artifacts saved in: {model_dir}")
            return model_dir, model_results, best_model_name

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise e

# try:
#     config = ConfigurationManager()
#     model_trainer_config = config.get_model_trainer_config()
#     model_trainer = ModelTrainer(config=model_trainer_config)
#     model_dir, results, best_model = model_trainer.train()
    
#     print(f"\nTraining completed! Models saved in: {model_dir}")
#     print(f"\nBest performing model: {best_model}")
#     print("\nModel Results:")
#     for model_name, result in results.items():
#         print(f"\n{model_name}:")
#         print(f"Best parameters: {result['best_params']}")
#         print(f"MSE Score: {result['best_score']}")
#         print(f"RMSE Score: {np.sqrt(result['best_score'])}")
    
# except Exception as e:
#     raise e