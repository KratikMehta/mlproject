import sys


def error_message_detail(error: Exception, error_detail=sys.exc_info()) -> str:
    _, _, exc_tb = error_detail
    if exc_tb is not None:
        filename = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    error_message = f"Error occured in python script {filename} on line {line_number}\nError message: {str(error)}"

    return error_message


class CustomException(Exception):
    def __init__(self, error: Exception, error_detail=sys.exc_info()) -> None:
        super().__init__(error)
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        return self.error_message
