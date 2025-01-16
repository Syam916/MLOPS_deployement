



from src.constants import *
from src.utils.common import *

from src.config.configuration import *

from src.entity.config_entity import *


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yml_file(config_filepath)
       
        self.schema = read_yml_file(schema_filepath)

        create_directories([self.config.artifacts_directory])
        

    def get_data_ingestion_config(self)->dataIngestionConfig:
        config=self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config=dataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_save=config.local_save,
            train_data_path= config.train_data_path,
            test_data_path= config.test_data_path
            
        )

        return data_ingestion_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            test_data_path=config.test_data_path,
            preprocessor_path=config.preprocessor_path,
            transformed_train_path=config.transformed_train_path,
            transformed_test_path=config.transformed_test_path,
            train_target_path=config.train_target_path,
            test_target_path=config.test_target_path
        )

        return data_transformation_config
    


    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            data_dir = config.data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

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