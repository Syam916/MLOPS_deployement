from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.config.configuration import ConfigurationManager
from src import logger

class Pipeline:
    def __init__(self):
        self.config = ConfigurationManager()

    def stage_01_data_ingestion(self):
        try:
            logger.info(f"\n\n{'='*20} Stage 1: Data Ingestion Started {'='*20}")
            data_ingestion_config = self.config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            train_data_path, test_data_path=data_ingestion.train_test_spliting()
            logger.info(f"{'='*20} Stage 1: Data Ingestion Completed {'='*20}")
            return train_data_path, test_data_path
        except Exception as e:
            logger.error(f"Error in data ingestion stage: {str(e)}")
            raise e

    def stage_02_data_validation(self):
        try:
            logger.info(f"\n\n{'='*20} Stage 2: Data Validation Started {'='*20}")
            data_validation_config = self.config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.validate_all_columns()
            logger.info(f"{'='*20} Stage 2: Data Validation Completed {'='*20}")
            return validation_status
        except Exception as e:
            logger.error(f"Error in data validation stage: {str(e)}")
            raise e

    def stage_03_data_transformation(self):
        try:
            logger.info(f"\n\n{'='*20} Stage 3: Data Transformation Started {'='*20}")
            data_transformation_config = self.config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation()
            logger.info(f"{'='*20} Stage 3: Data Transformation Completed {'='*20}")
            return train_arr, test_arr
        except Exception as e:
            logger.error(f"Error in data transformation stage: {str(e)}")
            raise e

    def stage_04_model_trainer(self):
        try:
            logger.info(f"\n\n{'='*20} Stage 4: Model Training Started {'='*20}")
            model_trainer_config = self.config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            best_model_path = model_trainer.train()
            logger.info(f"{'='*20} Stage 4: Model Training Completed {'='*20}")
            return best_model_path
        except Exception as e:
            logger.error(f"Error in model training stage: {str(e)}")
            raise e

    def stage_05_model_evaluation(self):
        try:
            logger.info(f"\n\n{'='*20} Stage 5: Model Evaluation Started {'='*20}")
            model_evaluation_config = self.config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            evaluation_results = model_evaluation.evaluate_model()
            logger.info(f"{'='*20} Stage 5: Model Evaluation Completed {'='*20}")
            return evaluation_results
        except Exception as e:
            logger.error(f"Error in model evaluation stage: {str(e)}")
            raise e

    def run_pipeline(self):
        """
        Run all stages of the pipeline in sequence
        """
        try:
            # Stage 1: Data Ingestion
            train_path, test_path = self.stage_01_data_ingestion()
            
            # Stage 2: Data Validation
            validation_status = self.stage_02_data_validation()
            if not validation_status:
                raise Exception("Data validation failed. Pipeline stopped.")
            
            # Stage 3: Data Transformation
            train_arr, test_arr = self.stage_03_data_transformation()
            
            # Stage 4: Model Training
            best_model_path = self.stage_04_model_trainer()
            
            # Stage 5: Model Evaluation
            evaluation_results = self.stage_05_model_evaluation()
            
            logger.info("Pipeline completed successfully!")
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise e