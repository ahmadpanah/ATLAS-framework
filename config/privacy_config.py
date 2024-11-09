
PRIVACY_CONFIG = {
    # Differential Privacy settings
    'epsilon': 1.0,  # Privacy budget
    'delta': 1e-5,   # Privacy failure probability
    'max_privacy_budget': 10.0,
    
    # Secure Multi-Party Computation settings
    'num_parties': 10,
    'threshold': 6,
    
    # Homomorphic Encryption settings
    'key_length': 2048,
    
    # Training settings
    'batch_size': 32,
    'learning_rate': 0.01,
    'num_epochs': 10,
    
    # Security settings
    'min_parties': 3,
    'max_retries': 3,
    'timeout': 300,  # seconds
    
    # Logging settings
    'log_level': 'INFO',
    'log_file': 'privacy.log'
}