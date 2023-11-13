import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

from sklearn.metrics import r2_score, mean_squared_error

from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, train_and_evaluate_models, choose_best_model

@dataclass
class ModelTrainerConfig:
    model_trainer_config_path = os.path.join("artifacts","model.pkl")

class ModerlTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def model_trainer(self, train_array, test_array):
        try:
            logging.info("Entered Model trainer")
            X_train, X_test, y_train, y_test = (
                train_array[:,:-1],
                test_array[:,:-1],
                train_array[:,-1],
                test_array[:,-1]
            )

            logging.info('Data Splittinf Successful')

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": xgb.XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }

            }

            report:dict = train_and_evaluate_models(X_train, X_test, y_train, y_test, models, params)

            best_model_name, best_score = choose_best_model(report)

            if best_score<0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset{best_model_name, best_score}")

            best_model = models[best_model_name]

            
            save_object(self.config.model_trainer_config_path,
                        best_model
                        )
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        
        except Exception as e:
            logging.error(CustomException(e,sys))
            CustomException(e,sys)
            