import os
from typing import Dict, Any

class BaseConfig:
    """Base configuration settings"""
    
    # General settings
    DEBUG = os.getenv('ATLAS_DEBUG', 'False').lower() == 'true'
    ENVIRONMENT = os.getenv('ATLAS_ENV', 'development')
    
    # Logging configuration
    LOG_LEVEL = os.getenv('ATLAS_LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = 'atlas.log'
    
    # Database configuration
    DB_HOST = os.getenv('ATLAS_DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('ATLAS_DB_PORT', '27017'))
    DB_NAME = os.getenv('ATLAS_DB_NAME', 'atlas_db')
    
    # API configuration
    API_VERSION = 'v1'
    API_PREFIX = f'/api/{API_VERSION}'
    API_HOST = os.getenv('ATLAS_API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('ATLAS_API_PORT', '8000'))
    
    # Component timeouts
    DEFAULT_TIMEOUT = 30  # seconds
    OPERATION_TIMEOUT = 300  # seconds
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get configuration dictionary"""
        return {
            'debug': cls.DEBUG,
            'environment': cls.ENVIRONMENT,
            'logging': {
                'level': cls.LOG_LEVEL,
                'format': cls.LOG_FORMAT,
                'file': cls.LOG_FILE
            },
            'database': {
                'host': cls.DB_HOST,
                'port': cls.DB_PORT,
                'name': cls.DB_NAME
            },
            'api': {
                'version': cls.API_VERSION,
                'prefix': cls.API_PREFIX,
                'host': cls.API_HOST,
                'port': cls.API_PORT
            },
            'timeouts': {
                'default': cls.DEFAULT_TIMEOUT,
                'operation': cls.OPERATION_TIMEOUT
            }
        }