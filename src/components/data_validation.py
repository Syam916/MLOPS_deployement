from dataclasses import dataclass
from pathlib import Path
import pandas as pd

@dataclass
class DataValidationConfig:
    root_dir:Path
    STATUS_FILE:str
    data_dir:Path
    all_schema:dict


from src.constants import *
from src.utils.common import *


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yml_file(config_filepath)
       
        self.schema = read_yml_file(schema_filepath)

        create_directories([self.config.artifacts_directory])

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
    


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")

            return validation_status
        
        except Exception as e:
            raise e

# try:
#     config = ConfigurationManager()
#     data_validation_config = config.get_data_validation_config()
#     data_validation = DataValiadtion(config=data_validation_config)
#     data_validation.validate_all_columns()
# except Exception as e:
#     raise e