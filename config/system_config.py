
from typing import Dict, Any
from .base_config import BaseConfig

class SystemConfig(BaseConfig):
    """System configuration"""
    
    # Resource limits
    RESOURCE_LIMITS = {
        'cpu_cores': 8,
        'memory': 16 * 1024 * 1024 * 1024,  # 16GB
        'disk_space': 1024 * 1024 * 1024 * 1024,  # 1TB
        'network_bandwidth': 1024 * 1024 * 1024  # 1Gbps
    }
    
    # Container settings
    CONTAINER_LIMITS = {
        'max_containers': 100,
        'cpu_shares': 1024,
        'memory_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'storage_limit': 10 * 1024 * 1024 * 1024  # 10GB
    }
    
    # Migration settings
    MIGRATION_CONFIG = {
        'max_parallel': 5,
        'chunk_size': 1024 * 1024,  # 1MB
        'retry_limit': 3,
        'timeout': 3600  # 1 hour
    }
    
    # Scheduling settings
    SCHEDULER_CONFIG = {
        'interval': 60,  # 1 minute
        'max_queue': 1000,
        'priority_levels': 5
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get system configuration"""
        config = super().get_config()
        config.update({
            'system': {
                'resources': cls.RESOURCE_LIMITS,
                'containers': cls.CONTAINER_LIMITS,
                'migration': cls.MIGRATION_CONFIG,
                'scheduler': cls.SCHEDULER_CONFIG
            }
        })
        return config