import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # Define the path for preprocessor.pkl file
    preprocessor_obj_file_path = os.path.join("artifact", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        # Read the preprocessor.pkl file path
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self) -> ColumnTransformer:
        try:
            # Define numerical and categorical columns
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Create numerical pipeline
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Create categorical pipeline
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler()),
                ]
            )

            # Create complete pipeline
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> "tuple[NDArray, NDArray, str]":
        try:
            # Read raw train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Succesfully imported raw train and test data")

            # Get preprocessing pipeline
            logging.info("Importing preprocessing object")
            preprocessor_obj = self.get_data_transformer_obj()

            # Defining input and target data
            target_column_name = "math_score"

            X_train_df = train_df.drop(columns=target_column_name)
            y_train_df = train_df[target_column_name]

            X_test_df = test_df.drop(columns=target_column_name)
            y_test_df = test_df[target_column_name]

            # Data transformation
            logging.info("Applying preprocessing pipeline to train and test data")

            X_train = preprocessor_obj.fit_transform(X_train_df)
            X_test = preprocessor_obj.transform(X_test_df)

            train_arr: NDArray = np.c_[X_train, np.array(y_train_df)]
            test_arr: NDArray = np.c_[X_test, np.array(y_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj,
            )
            logging.info(
                f"Saved pipeline to {self.data_transformation_config.preprocessor_obj_file_path} file"
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys.exc_info())
