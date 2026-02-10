"""
Settings Loader Module
Loads and validates configuration settings from YAML files
"""
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class SettingsLoader:
    """Load and manage application settings"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings loader
        
        Args:
            config_path: Path to settings YAML file
        """
        if config_path is None:
            config_path = "configs/settings.yaml"
        
        self.config_path = Path(config_path)
        self.settings = self._load_settings()
    
    def _load_settings(self) -> Dict[str, Any]:
        """
        Load settings from YAML file
        
        Returns:
            Dictionary containing settings
            
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If config file is invalid
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Settings file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                settings = yaml.safe_load(f)
            
            if settings is None:
                raise ValueError("Settings file is empty")
            
            # Validate critical settings
            self._validate_settings(settings)
            
            return settings
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing settings file: {e}")
    
    def _validate_settings(self, settings: Dict[str, Any]) -> None:
        """
        Validate required settings exist
        
        Args:
            settings: Settings dictionary to validate
            
        Raises:
            ValueError: If required settings are missing
        """
        required_sections = ['model', 'visualization', 'output']
        
        for section in required_sections:
            if section not in settings:
                raise ValueError(f"Required section '{section}' missing in settings")
        
        # Validate model section
        if 'weights_path' not in settings['model']:
            raise ValueError("'weights_path' missing in model settings")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get setting value by key (supports nested keys with dot notation)
        
        Args:
            key: Setting key (e.g., 'model.confidence_threshold')
            default: Default value if key not found
            
        Returns:
            Setting value or default
        """
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire settings section
        
        Args:
            section: Section name
            
        Returns:
            Settings section dictionary
        """
        return self.settings.get(section, {})
    
    def update(self, key: str, value: Any) -> None:
        """
        Update setting value
        
        Args:
            key: Setting key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        current = self.settings
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None) -> None:
        """
        Save settings to YAML file
        
        Args:
            output_path: Output file path (default: overwrite original)
        """
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.safe_dump(self.settings, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        return f"SettingsLoader(config_path='{self.config_path}')"


def load_settings(config_path: Optional[str] = None) -> SettingsLoader:
    """
    Convenience function to load settings
    
    Args:
        config_path: Path to settings file
        
    Returns:
        SettingsLoader instance
    """
    return SettingsLoader(config_path)
