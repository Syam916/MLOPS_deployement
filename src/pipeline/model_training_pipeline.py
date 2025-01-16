from src.config.configuration import ConfigurationManager
from src.components.model_trainer import ModelTrainer
from src import logger
import numpy as np

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            
            model_dir, results, best_model = model_trainer.train()
            
            logger.info(f"Training completed! Models saved in: {model_dir}")
            logger.info(f"Best performing model: {best_model}")
            
            for model_name, result in results.items():
                logger.info(f"\n{model_name}:")
                logger.info(f"Best parameters: {result['best_params']}")
                logger.info(f"MSE Score: {result['best_score']}")
                logger.info(f"RMSE Score: {np.sqrt(result['best_score'])}")
            
            logger.info("Model training completed")
            
            return model_dir, results, best_model
            
        except Exception as e:
            logger.error(f"Error in model training stage: {e}")
            raise e