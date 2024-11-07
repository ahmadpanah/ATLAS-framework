from typing import Dict, Any
from .base_config import BaseConfig

class MonitoringConfig(BaseConfig):
    """Monitoring configuration"""
    
    # Collection settings
    COLLECTION_INTERVAL = 1.0  # seconds
    WINDOW_SIZE = 100
    AGGREGATION_INTERVAL = 60  # seconds
    
    # Metrics configuration
    METRICS = {
        'system': [
            'cpu_usage',
            'memory_usage',
            'disk_usage',
            'network_io'
        ],
        'container': [
            'container_cpu',
            'container_memory',
            'container_network'
        ],
        'security': [
            'encryption_strength',
            'vulnerability_score',
            'threat_level'
        ],
        'performance': [
            'latency',
            'throughput',
            'response_time'
        ]
    }
    
    # Alerting thresholds
    THRESHOLDS = {
        'cpu_usage': 80.0,
        'memory_usage': 85.0,
        'disk_usage': 90.0,
        'latency': 100.0,  # ms
        'threat_level': 0.7
    }
    
    # Storage settings
    RETENTION_PERIOD = 7 * 24 * 60 * 60  # 7 days
    MAX_SAMPLES = 1000000
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get monitoring configuration"""
        config = super().get_config()
        config.update({
            'monitoring': {
                'collection': {
                    'interval': cls.COLLECTION_INTERVAL,
                    'window_size': cls.WINDOW_SIZE,
                    'aggregation_interval': cls.AGGREGATION_INTERVAL
                },
                'metrics': cls.METRICS,
                'thresholds': cls.THRESHOLDS,
                'storage': {
                    'retention_period': cls.RETENTION_PERIOD,
                    'max_samples': cls.MAX_SAMPLES
                }
            }
        })
        return config