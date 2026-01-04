import os 
import sys 

import pandas as pd
import numpy as np

import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save a Python object to a file using pickle.

    Args:
        file_path (str): The path where the object should be saved.
        obj: The Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models , params):
    """
    Evaluate multiple machine learning models and return their R2 scores.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training target.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing target.
        models (dict): A dictionary where keys are model names and values are model instances.

    Returns:
        dict: A dictionary with model names as keys and their R2 scores as values.
    """
    try:
       report ={}

       for i in range(len(list(models))):
            model = list(models.values())[i]
            params = params[list(models.keys())[i]]

            gs= GridSearchCV(model,params,cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)       
            model.fit(X_train, y_train)       

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score =r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

       return report

    except Exception as e:
        raise CustomException(e, sys)