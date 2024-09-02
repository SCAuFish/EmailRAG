import logging


def activate_logger(name="default", level=logging.INFO,
                    format="%(asctime)s [%(module)s.%(funcName)s:%(lineno)d] %(levelname)s: %(message)s"):
    logger = logging.getLogger(name)
    logger.propagate = False  # prevents double log entries
    formatter = logging.Formatter(format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def construct_content(from_name: str, to_name: str, email_content: str) -> str:
    return f"Email from {from_name} to {to_name} below\n{email_content}"
