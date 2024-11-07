from typing import Dict, Any
from .base_config import BaseConfig

class SecurityConfig(BaseConfig):
    """Security configuration"""
    
    # Security levels
    SECURITY_LEVELS = {
        'LOW': {
            'encryption': 'AES-128-GCM',
            'key_rotation': 86400,  # 24 hours
            'monitoring': 'basic'
        },
        'MEDIUM': {
            'encryption': 'AES-256-GCM',
            'key_rotation': 43200,  # 12 hours
            'monitoring': 'enhanced'
        },
        'HIGH': {
            'encryption': 'AES-256-GCM',
            'key_rotation': 21600,  # 6 hours
            'monitoring': 'continuous'
        },
        'CRITICAL': {
            'encryption': 'ChaCha20-Poly1305',
            'key_rotation': 3600,  # 1 hour
            'monitoring': 'real-time'
        }
    }
    
    # Authentication settings
    AUTH_TOKEN_EXPIRY = 3600  # 1 hour
    MAX_LOGIN_ATTEMPTS = 3
    PASSWORD_MIN_LENGTH = 12
    
    # Audit settings
    AUDIT_LOG_FILE = 'security_audit.log'
    AUDIT_RETENTION = 90  # days
    
    # Threat detection
    THREAT_CHECK_INTERVAL = 300  # 5 minutes
    MAX_THREAT_LEVEL = 0.8
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get security configuration"""
        config = super().get_config()
        config.update({
            'security': {
                'levels': cls.SECURITY_LEVELS,
                'authentication': {
                    'token_expiry': cls.AUTH_TOKEN_EXPIRY,
                    'max_login_attempts': cls.MAX_LOGIN_ATTEMPTS,
                    'password_min_length': cls.PASSWORD_MIN_LENGTH
                },
                'audit': {
                    'log_file': cls.AUDIT_LOG_FILE,
                    'retention': cls.AUDIT_RETENTION
                },
                'threat_detection': {
                    'check_interval': cls.THREAT_CHECK_INTERVAL,
                    'max_level': cls.MAX_THREAT_LEVEL
                }
            }
        })
        return config