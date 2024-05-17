import os
import sys

import dill

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
