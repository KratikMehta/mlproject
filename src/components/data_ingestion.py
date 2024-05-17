import os
import sys
from dataclasses import dataclass
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    # Define the paths for output files
    train_data_path: str = os.path.join("artifact", "train.csv")
    test_data_path: str = os.path.join("artifact", "test.csv")
    raw_data_path: str = os.path.join("artifact", "data.csv")


class DataIngestion:
    def __init__(self) -> None:
        # Read the output file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingesion(self) -> "tuple[str, str]":
        logging.info("Started data ingestion process")
        try:
            # Read the dataset
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Read the dataset as dataframe")

            # Make the output files directory
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            # Save the raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the data into train and test set
            logging.info("train_test_spilt initiated")
            split_data: List[pd.DataFrame] = train_test_split(
                df, test_size=0.2, random_state=42
            )
            train_set, test_set = split_data

            # Save the train and test set
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Data Ingestion Completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys.exc_info())

if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingesion()
    
    transform_obj = DataTransformation()
    transform_obj.initiate_data_transformation(train_path, test_path)