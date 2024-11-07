from typing import Dict, Any
from .base_config import BaseConfig

class NetworkConfig(BaseConfig):
    """Network configuration"""
    
    # Network monitoring
    PING_INTERVAL = 1.0  # seconds
    PING_TIMEOUT = 1.0  # seconds
    PING_COUNT = 5
    
    # Bandwidth settings
    MIN_BANDWIDTH = 10 * 1024 * 1024  # 10 MB/s
    MAX_BANDWIDTH = 1024 * 1024 * 1024  # 1 GB/s
    
    # Quality thresholds
    LATENCY_THRESHOLD = 100.0  # ms
    PACKET_LOSS_THRESHOLD = 0.01  # 1%
    JITTER_THRESHOLD = 10.0  # ms
    
    # Connection settings
    CONNECTION_TIMEOUT = 30  # seconds
    MAX_RETRIES = 3
    RETRY_INTERVAL = 5  # seconds
    
    # Protocol settings
    PROTOCOLS = {
        'data': 'TCP',
        'control': 'TCP',
        'monitoring': 'UDP'
    }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get network configuration"""
        config = super().get_config()
        config.update({
            'network': {
                'monitoring': {
                    'ping_interval': cls.PING_INTERVAL,
                    'ping_timeout': cls.PING_TIMEOUT,
                    'ping_count': cls.PING_COUNT
                },
                'bandwidth': {
                    'min': cls.MIN_BANDWIDTH,
                    'max': cls.MAX_BANDWIDTH
                },
                'thresholds': {
                    'latency': cls.LATENCY_THRESHOLD,
                    'packet_loss': cls.PACKET_LOSS_THRESHOLD,
                    'jitter': cls.JITTER_THRESHOLD
                },
                'connection': {
                    'timeout': cls.CONNECTION_TIMEOUT,
                    'max_retries': cls.MAX_RETRIES,
                    'retry_interval': cls.RETRY_INTERVAL
                },
                'protocols': cls.PROTOCOLS
            }
        })
        return config