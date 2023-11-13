import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModerlTrainer


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting Ingestion Component")
        try:
            
            df = pd.read_csv('D:\\ML_Project\\data\\test.csv')
            
            logging.info('Read the data from source into DataFrame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Splitting the data into train and test')

            train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_data.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Ingestion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.error(CustomException(e,sys))
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    di = DataIngestion()
    train_data, test_data = di.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, X_test, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModerlTrainer()
    model_trainer.model_trainer(X_train,X_test)