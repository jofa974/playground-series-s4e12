import logging


def setup_logger(log_level: int = logging.DEBUG, name: str | None = None):
    """
    Sets up a logger that logs messages to both the terminal and a file.

    Args:
        log_file (str): Path to the log file. If None, the log file will be named with the current date and time.
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create a logger
    if name is None:
        name = "logger"
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create handlers
    console_handler = logging.StreamHandler()

    # Set log levels for handlers
    console_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Add formatter to handlers
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    return logger
