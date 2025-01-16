from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from box.exceptions import BoxValueError
import yaml,os
from src import logger

@ensure_annotations
def read_yml_file(file_path:Path) -> ConfigBox:
    """Read a yml file and return a ConfigBox object"""


    try:
        with open(file_path,'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {file_path} loaded successfully")
            return ConfigBox(content)

    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories : list,verbose=True):
    for dir in path_to_directories:
        os.makedirs(dir,exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {dir}")