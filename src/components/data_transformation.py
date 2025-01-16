from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
from src.constants import *
from src.utils.common import *

@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    train_target_path: Path
    test_target_path: Path

class ConfigurationManager:
    def __init__(self, config=CONFIG_FILE_PATH):
        self.config = read_yml_file(CONFIG_FILE_PATH)
        create_directories([self.config.artifacts_directory])

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

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        
    def get_data_transformer(self, train_data):
        """
        Create preprocessing pipeline for numerical and categorical data
        """
        # Separate numerical and categorical columns
        numerical_columns = train_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = train_data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target variable from numerical columns
        if 'median_house_value' in numerical_columns:
            numerical_columns.remove('median_house_value')
        
        # Create preprocessing pipelines
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(drop='first'))
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('num_transform', num_pipeline, numerical_columns),
            ('cat_transform', cat_pipeline, categorical_columns)
        ])
        
        return preprocessor
    
    def transform_data(self):
        """
        Transform the training and test data
        """
        try:
            # Load the training and test data
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)
            
            logger.info("Loaded train and test data")
            
            # Create preprocessor
            preprocessor = self.get_data_transformer(train_data)
            
            # Separate features and target
            X_train = train_data.drop('median_house_value', axis=1)
            y_train = train_data['median_house_value']
            X_test = test_data.drop('median_house_value', axis=1)
            y_test = test_data['median_house_value']
            
            logger.info("Split data into features and target")
            
            # Fit and transform training data
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)
            
            logger.info("Transformed training and test data")
            
            # Save preprocessor
            with open(self.config.preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            # Save transformed data
            np.save(self.config.transformed_train_path, X_train_transformed)
            np.save(self.config.transformed_test_path, X_test_transformed)
            np.save(self.config.train_target_path, y_train)
            np.save(self.config.test_target_path, y_test)
            
            logger.info("Saved preprocessor and transformed data")
            
            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test
            )
            
        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise e

# try:
#     config = ConfigurationManager()
#     data_transformation_config = config.get_data_transformation_config()
#     data_transformation = DataTransformation(config=data_transformation_config)
    
#     X_train_transformed, X_test_transformed, y_train, y_test = data_transformation.transform_data()
    
#     print("Transformed training data shape:", X_train_transformed.shape)
#     print("Transformed test data shape:", X_test_transformed.shape)
    
# except Exception as e:
#     raise e