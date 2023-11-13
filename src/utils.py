import os
import sys
import dill
from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def train_and_evaluate_models(X_train, X_test, y_train, y_test, models, params):
    report = {}
    logging.info('Train and Evaluation of model started')
    for model_name, model in models.items():
        param = params[model_name]

        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train, y_train)

        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        report[model_name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
        
    return report

def choose_best_model(report):
    logging.info('Choosing the best model based on r2_score')
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, metrics in report.items():
        r2 = metrics['R2']
        if r2 > best_r2:
            best_model = model_name
            best_r2 = r2
    
    return best_model, best_r2



