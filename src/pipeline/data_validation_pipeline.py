# src/pipeline/stage_02_data_validation.py
from src.config.configuration import ConfigurationManager
from src.components.data_validation import DataValidation
from src import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.validate_all_columns()
            
            logger.info(f"Data validation completed with status: {validation_status}")
            
            return validation_status
            
        except Exception as e:
            logger.error(f"Error in data validation stage: {e}")
            raise e