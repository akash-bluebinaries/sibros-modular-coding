from src.config import *
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *
from src.utils import *

import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "XGBRegressor": XGBClassifier(),
                "DecisionTreeRegressor": DecisionTreeClassifier(),
                "GradientBoostingRegressor": GradientBoostingClassifier(),
                "RandomForestRegressor": RandomForestClassifier(),
                "SVC": SVC()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(f"Best model found: {best_model}, Accuracy: {best_model_score}")
            logging.info(f"Best model found: {best_model}, Accuracy: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj= best_model)


        except Exception as e:
            raise CustomException(e,sys)