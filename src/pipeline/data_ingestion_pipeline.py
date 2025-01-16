# src/pipeline/stage_01_data_ingestion.py
from src.config.configuration import ConfigurationManager
from src.components.data_ingestion import DataIngestion
from src import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
        
    def main(self):
        try:
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.train_test_spliting()
            logger.info("Data ingestion completed")
            
        except Exception as e:
            logger.error(f"Error in data ingestion stage: {e}")
            raise e