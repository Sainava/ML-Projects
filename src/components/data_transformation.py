import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

# To define the output path of the preprocessor object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    '''
    Docstring for DataTransformation

    This class is responsible for handling all data transformation tasks including:
    - Handling missing values
    - Scaling numerical features
    - Encoding categorical features
    - Combining these transformations into a single preprocessor object
    '''

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")

            # Numerical and Categorical columns definition
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            #handling missing values and scaling for numerical columns
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), #handling missing values (median as outliers exist)
                ('scaler', StandardScaler())
            ])
            #handling missing values and encoding for categorical columns
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), #handling missing values via mode
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False)) # Optional: Scaling after one-hot encoding
            ])

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            #Combining both numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor 
        
        except Exception as e:
            logging.error(f"Error in data transformation: {e}")
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            #Reading the train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            #Separating features and target variable from training and testing data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            #Transforming the input features using preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            #Combining transformed features and target variable into a single array
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            #Saving the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor object saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {e}")
            raise CustomException(e, sys)