import logging
from datetime import datetime


def setup_logger(log_file: str = None, log_level: int = logging.DEBUG):
    """
    Sets up a logger that logs messages to both the terminal and a file.

    Args:
        log_file (str): Path to the log file. If None, the log file will be named with the current date and time.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # If no log file is provided, create one with the current date and time
    if log_file is None:
        log_file = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")

    # Create a logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(log_level)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    # Set log levels for handlers
    console_handler.setLevel(log_level)
    file_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
