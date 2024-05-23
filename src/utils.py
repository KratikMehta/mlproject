import os
import sys
from typing import Any

import dill
from numpy.typing import NDArray
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.exception import CustomException


def save_object(file_path: str, obj) -> None:
    try:
        # Get directory path of the file
        dir_path = os.path.dirname(file_path)
        # Create the directory
        os.makedirs(dir_path, exist_ok=True)
        # Dump the obj to a file
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys.exc_info())


def evaluate_models(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    models: "dict[str, Pipeline]",
    params: "dict[str, dict[str, Any]]",
) -> "dict[str, float]":
    try:
        report: dict[str, float] = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model_name = list(models.keys())[i]
            param = list(params.values())[i]

            # perform grid search
            gs = GridSearchCV(model, param, n_jobs=-1, cv=3)
            gs.fit(X_train, y_train)

            # Best model
            model = gs.best_estimator_
            # Make predictions
            y_test_pred = model.predict(X_test)
            # Evaluate Test dataset
            test_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = float(test_score)
            models[model_name] = model
        return report
    except Exception as e:
        raise CustomException(e, sys.exc_info())


def load_object(file_path) -> Any:
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception as e:
        CustomException(e, sys.exc_info())
