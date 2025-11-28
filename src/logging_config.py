import logging
import sys
import os
import warnings
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv
from colorlog import ColoredFormatter

load_dotenv()


def setup_logging():
    """
    Configures global logging for the entire application.
    Must be called once at the app entry point, before any loggers are created.
    """
    # 1. Call the warning filter function FIRST
    filter_warnings()

    root = logging.getLogger()
    if root.handlers:
        return

    ENV = os.getenv("ENV", "dev").lower()
    # level = logging.DEBUG if ENV == "dev" else logging.INFO
    level = logging.INFO
    root.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)

    # DEV mode console logs
    if ENV == "dev":

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

    # In production, go with JSON.
    else:
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"
        )

    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.propagate = False
    
    suppress_noisy_logs()

    root.info(f"Logging initialized (mode={ENV}, level={level})")

def suppress_noisy_logs():
    noisy = [
        "googleapiclient",
        "googleapiclient.discovery_cache",
        "urllib3",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.ERROR)


def filter_warnings():
    warnings.filterwarnings(
        "ignore", 
        message=r"You are using a Python version.*", 
        category=FutureWarning, 
        module=r"google\.api_core\._python_version_support"
    )