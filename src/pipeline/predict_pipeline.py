import os
from typing import Union

import pandas as pd
from sklearn.pipeline import Pipeline

from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        self.model_path = os.path.join("artifact", "model.pkl")
        self.preprocessor_path = os.path.join("artifact", "preprocessor.pkl")

    def predict(
        self, features: pd.DataFrame
    ):  # -> ndarray[Any, Any] | tuple[ndarray[Any, Any], ndarray[Any,...:
        try:
            model: Pipeline = load_object(self.model_path)
            preprocessor: Pipeline = load_object(self.preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            return e


class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ) -> None:
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self) -> Union[pd.DataFrame, Exception]:
        try:
            custom_data_dict = {
                "gender": self.gender,
                "race_ethnicity": self.race_ethnicity,
                "parental_level_of_education": self.parental_level_of_education,
                "lunch": self.lunch,
                "test_preparation_course": self.test_preparation_course,
                "reading_score": self.reading_score,
                "writing_score": self.writing_score,
            }
            return pd.DataFrame(custom_data_dict, index=[0])
        except Exception as e:
            return e
