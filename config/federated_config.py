
from typing import Dict, Any
from .base_config import BaseConfig

class FederatedConfig(BaseConfig):
    """Federated learning configuration"""
    
    # Training settings
    MIN_CLIENTS = 3
    MAX_CLIENTS = 10
    ROUNDS = 100
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    
    # Model architecture
    MODEL_CONFIG = {
        'hidden_layers': [512, 256, 128],
        'activation': 'relu',
        'dropout_rate': 0.2
    }
    
    # Privacy settings
    PRIVACY_BUDGET = 1.0
    NOISE_MULTIPLIER = 1.1
    MAX_GRAD_NORM = 1.0
    
    # Aggregation settings
    AGGREGATION_METHOD = 'FedAvg'
    MIN_SAMPLES_PER_CLIENT = 100
    
    # Communication settings
    TIMEOUT = 300  # seconds
    MAX_RETRIES = 3
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get federated learning configuration"""
        config = super().get_config()
        config.update({
            'federated_learning': {
                'training': {
                    'min_clients': cls.MIN_CLIENTS,
                    'max_clients': cls.MAX_CLIENTS,
                    'rounds': cls.ROUNDS,
                    'local_epochs': cls.LOCAL_EPOCHS,
                    'batch_size': cls.BATCH_SIZE,
                    'learning_rate': cls.LEARNING_RATE
                },
                'model': cls.MODEL_CONFIG,
                'privacy': {
                    'budget': cls.PRIVACY_BUDGET,
                    'noise_multiplier': cls.NOISE_MULTIPLIER,
                    'max_grad_norm': cls.MAX_GRAD_NORM
                },
                'aggregation': {
                    'method': cls.AGGREGATION_METHOD,
                    'min_samples': cls.MIN_SAMPLES_PER_CLIENT
                },
                'communication': {
                    'timeout': cls.TIMEOUT,
                    'max_retries': cls.MAX_RETRIES
                }
            }
        })
        return config