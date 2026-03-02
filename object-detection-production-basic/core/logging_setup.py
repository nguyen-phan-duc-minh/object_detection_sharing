import yaml
from pathlib import Path
import logging.config

def setup_logging():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    with open("config/logging.yaml") as file:
        logging.config.dictConfig(yaml.safe_load(file))