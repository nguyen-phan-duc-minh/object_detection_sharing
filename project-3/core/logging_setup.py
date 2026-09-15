import logging.config
import yaml
from pathlib import Path

def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    with open("config/logging.yaml") as file:
        logging.config.dictConfig(yaml.safe_load(file))
