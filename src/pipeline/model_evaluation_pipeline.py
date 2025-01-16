from src.config.configuration import ConfigurationManager
from src.components.model_evaluation import ModelEvaluation
from src import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            
            evaluation_results = model_evaluation.evaluate_model()
            
            logger.info("\nModel Performance Metrics:")
            logger.info(f"R² Score: {evaluation_results['R2_Score']:.4f}")
            logger.info(f"RMSE: ${evaluation_results['RMSE']:,.2f}")
            logger.info(f"MAE: ${evaluation_results['MAE']:,.2f}")
            logger.info(f"Model Type: {evaluation_results['Model_Type']}")
            
            logger.info("Model evaluation completed")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in model evaluation stage: {e}")
            raise e