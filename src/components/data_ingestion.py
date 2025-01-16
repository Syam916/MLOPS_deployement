from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

@dataclass
class dataIngestionConfig:
    root_dir: Path
    source_url: str
    local_save : Path

from src.constants import *
from src.utils.common import *

class ConfigurationManager:
    def __init__(self,config=CONFIG_FILE_PATH):
        self.config=read_yml_file(CONFIG_FILE_PATH)

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
    
    

class DataIngestion:
    def __init__(self,config:dataIngestionConfig):
        self.config=config

    def download_file(self):

        if not os.path.exists(self.config.local_save):
            df=pd.read_csv(self.config.source_url)

            df.to_csv(self.config.local_save,index=False)
            logger.info("{self.config.loacal_save} is craeted")

        else:
            logger.info('data file already exists')

    def train_test_spliting(self):
        data = pd.read_csv(self.config.local_save)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        train.to_csv(self.config.train_data_path,index = False)
        test.to_csv(self.config.test_data_path,index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        return self.config.train_data_path,self.config.test_data_path,

# try:
#     config= ConfigurationManager()
#     data_ingestion_config=config.get_data_ingestion_config()
#     data_ingestion=DataIngestion(config=data_ingestion_config)

#     data_ingestion.download_file()
# except Exception as e:
#     raise e