from typing import Dict, Any
from .base_config import BaseConfig

class EncryptionConfig(BaseConfig):
    """Encryption-specific configuration"""
    
    # Algorithm configurations
    ENCRYPTION_ALGORITHMS = {
        'AES-256-GCM': {
            'key_size': 256,
            'mode': 'GCM',
            'nonce_size': 12,
            'tag_size': 16,
            'priority': 1
        },
        'CHACHA20-POLY1305': {
            'key_size': 256,
            'nonce_size': 12,
            'tag_size': 16,
            'priority': 2
        },
        'AES-256-CBC': {
            'key_size': 256,
            'mode': 'CBC',
            'iv_size': 16,
            'priority': 3
        }
    }
    
    # Key derivation settings
    KDF_ITERATIONS = 200000
    KDF_ALGORITHM = 'PBKDF2-SHA256'
    SALT_SIZE = 16
    
    # Key management
    KEY_ROTATION_INTERVAL = 86400  # 24 hours
    KEY_CACHE_SIZE = 1000
    MASTER_KEY_SIZE = 32
    
    # Performance settings
    BUFFER_SIZE = 1024 * 1024  # 1MB
    MAX_MEMORY = 1024 * 1024 * 1024  # 1GB
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get encryption configuration"""
        config = super().get_config()
        config.update({
            'encryption': {
                'algorithms': cls.ENCRYPTION_ALGORITHMS,
                'kdf': {
                    'iterations': cls.KDF_ITERATIONS,
                    'algorithm': cls.KDF_ALGORITHM,
                    'salt_size': cls.SALT_SIZE
                },
                'key_management': {
                    'rotation_interval': cls.KEY_ROTATION_INTERVAL,
                    'cache_size': cls.KEY_CACHE_SIZE,
                    'master_key_size': cls.MASTER_KEY_SIZE
                },
                'performance': {
                    'buffer_size': cls.BUFFER_SIZE,
                    'max_memory': cls.MAX_MEMORY
                }
            }
        })
        return config