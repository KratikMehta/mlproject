import sys

from src import logger

logging = logger.logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script {filename} on line {exc_tb.tb_lineno}\nError message: {str(error)}"

    return error_message


class CustomException(Exception):
    def __init__(self, error: Exception, error_detail: sys) -> None:
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
