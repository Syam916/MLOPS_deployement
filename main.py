# main.py
from src.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.pipeline.model_training_pipeline import ModelTrainerTrainingPipeline
from src.pipeline.model_evaluation_pipeline import ModelEvaluationTrainingPipeline
from src import logger

def main():
    try:
        # Stage 1: Data Ingestion
        logger.info(">>>>>> Stage 1: Data Ingestion Started <<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(">>>>>> Stage 1: Data Ingestion Completed <<<<<<\n\nx==========x")
        
        # Stage 2: Data Validation
        logger.info(">>>>>> Stage 2: Data Validation Started <<<<<<")
        data_validation = DataValidationTrainingPipeline()
        validation_status = data_validation.main()
        
        if not validation_status:
            logger.error("Data validation failed. Stopping pipeline.")
            raise ValueError("Data validation failed")
            
        logger.info(">>>>>> Stage 2: Data Validation Completed <<<<<<\n\nx==========x")
        
        # Stage 3: Data Transformation
        logger.info(">>>>>> Stage 3: Data Transformation Started <<<<<<")
        data_transformation = DataTransformationTrainingPipeline()
        X_train_transformed, X_test_transformed, y_train, y_test = data_transformation.main()
        logger.info(">>>>>> Stage 3: Data Transformation Completed <<<<<<\n\nx==========x")
        
        # Stage 4: Model Training
        logger.info(">>>>>> Stage 4: Model Training Started <<<<<<")
        model_trainer = ModelTrainerTrainingPipeline()
        model_dir, results, best_model = model_trainer.main()
        logger.info(">>>>>> Stage 4: Model Training Completed <<<<<<\n\nx==========x")
        
        # Stage 5: Model Evaluation
        logger.info(">>>>>> Stage 5: Model Evaluation Started <<<<<<")
        model_evaluation = ModelEvaluationTrainingPipeline()
        evaluation_results = model_evaluation.main()
        logger.info(">>>>>> Stage 5: Model Evaluation Completed <<<<<<\n\nx==========x")
        
    except Exception as e:
        logger.exception(e)
        raise e

if __name__ == "__main__":
    main()