# src/pipeline/stage_02_data_transformation.py
from src.config.configuration import ConfigurationManager
from src.components.data_transformation import DataTransformation
from src import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            
            X_train_transformed, X_test_transformed, y_train, y_test = data_transformation.transform_data()
            
            logger.info(f"Transformed training data shape: {X_train_transformed.shape}")
            logger.info(f"Transformed test data shape: {X_test_transformed.shape}")
            logger.info("Data transformation completed")
            
            return X_train_transformed, X_test_transformed, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in data transformation stage: {e}")
            raise e