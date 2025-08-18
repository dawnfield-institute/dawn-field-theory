"""Configuration management."""


import yaml

class ConfigManager:
    """
    Handles configuration loading and management for the symbolic engine.
    """
    def __init__(self, config_path: str = None):
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, path: str):
        """
        Load configuration from a YAML file.
        """
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)
