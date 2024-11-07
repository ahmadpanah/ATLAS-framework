from typing import Dict, Any
from .base_config import BaseConfig
from .encryption_config import EncryptionConfig
from .federated_config import FederatedConfig
from .monitoring_config import MonitoringConfig
from .network_config import NetworkConfig
from .security_config import SecurityConfig
from .system_config import SystemConfig

class AtlasConfig:
    """ATLAS framework configuration"""
    
    @staticmethod
    def load_config() -> Dict[str, Any]:
        """Load complete configuration"""
        config = {}
        
        # Load configurations from all components
        config.update(BaseConfig.get_config())
        config.update(EncryptionConfig.get_config())
        config.update(FederatedConfig.get_config())
        config.update(MonitoringConfig.get_config())
        config.update(NetworkConfig.get_config())
        config.update(SecurityConfig.get_config())
        config.update(SystemConfig.get_config())
        
        return config

# Default configuration instance
default_config = AtlasConfig.load_config()