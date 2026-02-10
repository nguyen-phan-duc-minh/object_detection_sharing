"""
Logging Setup Module
Configures logging for the entire application
"""
import logging
import logging.config
from pathlib import Path
from typing import Optional

import yaml


class LoggingSetup:
    """Setup and configure logging for the application"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize logging setup
        
        Args:
            config_path: Path to logging configuration YAML file
        """
        if config_path is None:
            config_path = "configs/logging.yaml"
        
        self.config_path = Path(config_path)
        self._ensure_log_directory()
        self._setup_logging()
    
    def _ensure_log_directory(self) -> None:
        """Ensure log directory exists"""
        log_dir = Path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logging.config.dictConfig(config)
                print(f"Logging configured from: {self.config_path}")
            else:
                # Fallback to basic config
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                print(f"Warning: Logging config not found at {self.config_path}, using basic config")
        except Exception as e:
            print(f"Error setting up logging: {e}")
            logging.basicConfig(level=logging.INFO)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a logger instance
        
        Args:
            name: Logger name
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)


def setup_logging(config_path: Optional[str] = None) -> None:
    """
    Convenience function to setup logging
    
    Args:
        config_path: Path to logging configuration file
    """
    LoggingSetup(config_path)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return LoggingSetup.get_logger(name)
