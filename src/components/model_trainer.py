import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from numpy.typing import NDArray
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    # Define the path for model.pkl file
    model_file_path = os.path.join("artifact", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self, train_array: NDArray, test_array: NDArray
    ):  # -> Float | ndarray[Any, Any]:
        try:
            logging.info("Splitting train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define the models
            models = {
                "Random Forest Regressor": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Get the models report
            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )

            # Get the best model
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(Exception("No best model found"), sys.exc_info())
            logging.info(f"Best model: {best_model}")

            # Saving the model
            save_object(self.model_trainer_config.model_file_path, best_model)

            # Find prediction score
            predictions = best_model.predict(X_test)
            r2_squared = r2_score(y_test, predictions)

            return r2_squared
        except Exception as e:
            raise CustomException(e, sys.exc_info())
