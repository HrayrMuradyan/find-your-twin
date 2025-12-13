import logging
import sys
from dotenv import load_dotenv
from colorlog import ColoredFormatter

load_dotenv()

def setup_logging():
    """
    Configures global logging for the entire application.
    Must be called once at the app entry point, before any loggers are created.
    """

    root = logging.getLogger()
    if root.handlers:
        return

    level = logging.INFO
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)

    LOG_FORMAT = (
        "%(asctime)s | "
        "%(log_color)s%(levelname)-8s%(reset)s | "
        "%(name)s | "
        "%(message)s"
    )
    
    formatter = ColoredFormatter(
        LOG_FORMAT,
        datefmt="%y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.propagate = False
    
    suppress_noisy_logs()

    root.info(f"Logging initialized (level={level})")

def suppress_noisy_logs():
    noisy = [
        "googleapiclient",
        "googleapiclient.discovery_cache",
        "urllib3",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.ERROR)
