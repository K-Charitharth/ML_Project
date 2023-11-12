import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformation(self):

        try:
            cat_col = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_col = ["reading_score","writing_score"]

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ('scaler', StandardScaler())

                ]
            )

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())

                ]
            )

            logging.info("Pipelines created for categorical and numerical features")

            preprocessor = ColumnTransformer(
                [
                    ("cat_pipeline", cat_pipeline, cat_col),
                    ("num_pipeline", num_pipeline, num_col)
                ]
            )

            logging.info("preprocessing done using column transformer")

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test DataFrames completed')

            prprocessing_obj = self.get_data_tranformation()

            target_col = "math_score"
            num_cols = ["reading_score","writing_score"]

            input_feature_train_df = train_df.drop(columns=[target_col])
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=[target_col])
            target_feature_test_df = test_df[target_col]

            train_arr = np.c_[
                input_feature_train_df, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_df, np.array(target_feature_test_df)
            ]

            logging.info('crated train and test arrays')

            save_object(

                file_path = self.data_transformation_config.preprocessor_file_path,
                obj = prprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)