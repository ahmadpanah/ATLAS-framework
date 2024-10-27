import os
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
from datetime import datetime
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from threading import Lock
from ..utils.data_structures import SecurityLevel, NetworkMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdaptiveEncryptionEngine:
    """
    Implements the Adaptive Encryption Engine component from the paper.
    Provides dynamic encryption parameter selection and data encryption/decryption.
    """
    def __init__(self):
        self.lock = Lock()
        self.active_keys = {}
        self.encryption_stats = {}
        
        # Define encryption configurations for different security levels
        self.encryption_configs = {
            SecurityLevel.LOW: {
                'algorithm': 'AES',
                'key_size': 128,
                'mode': 'CBC',
                'iterations': 100000,
                'memory_hard': False
            },
            SecurityLevel.MEDIUM: {
                'algorithm': 'AES',
                'key_size': 192,
                'mode': 'CBC',
                'iterations': 150000,
                'memory_hard': False
            },
            SecurityLevel.HIGH: {
                'algorithm': 'AES',
                'key_size': 256,
                'mode': 'GCM',
                'iterations': 200000,
                'memory_hard': True
            },
            SecurityLevel.CRITICAL: {
                'algorithm': 'AES',
                'key_size': 256,
                'mode': 'GCM',
                'iterations': 250000,
                'memory_hard': True,
                'additional_protection': True
            }
        }

    def adapt_encryption(self, 
                        container_id: str, 
                        security_level: SecurityLevel,
                        network_metrics: NetworkMetrics) -> Dict:
        """
        Adapt encryption parameters based on security requirements and network conditions
        Implements Algorithm 3 from the paper
        """
        try:
            with self.lock:
                # Get base configuration for security level
                base_config = self.encryption_configs[security_level].copy()
                
                # Adjust parameters based on network conditions
                adjusted_config = self._adjust_parameters(base_config, network_metrics)
                
                # Generate new encryption key
                key, salt = self._generate_key(adjusted_config)
                
                # Store active key and configuration
                self.active_keys[container_id] = {
                    'key': key,
                    'salt': salt,
                    'config': adjusted_config,
                    'created_at': datetime.now()
                }
                
                # Return configuration (without sensitive key material)
                return {
                    'container_id': container_id,
                    'algorithm': adjusted_config['algorithm'],
                    'key_size': adjusted_config['key_size'],
                    'mode': adjusted_config['mode'],
                    'memory_hard': adjusted_config['memory_hard'],
                    'parameters': self._get_public_parameters(adjusted_config)
                }
                
        except Exception as e:
            logger.error(f"Encryption adaptation failed: {str(e)}")
            raise

    def encrypt_data(self, container_id: str, data: bytes) -> Dict:
        """Encrypt data using adapted parameters"""
        try:
            with self.lock:
                if container_id not in self.active_keys:
                    raise ValueError(f"No encryption configuration for container {container_id}")
                
                key_info = self.active_keys[container_id]
                key = key_info['key']
                config = key_info['config']
                
                # Generate IV
                iv = os.urandom(16)
                
                # Create cipher based on configuration
                if config['mode'] == 'GCM':
                    cipher = Cipher(
                        algorithms.AES(key),
                        modes.GCM(iv)
                    )
                else:  # CBC mode
                    cipher = Cipher(
                        algorithms.AES(key),
                        modes.CBC(iv)
                    )
                
                encryptor = cipher.encryptor()
                
                # Add padding if needed
                if config['mode'] == 'CBC':
                    padder = padding.PKCS7(128).padder()
                    padded_data = padder.update(data) + padder.finalize()
                else:
                    padded_data = data
                
                # Encrypt data
                ciphertext = encryptor.update(padded_data) + encryptor.finalize()
                
                # Prepare result
                result = {
                    'iv': iv,
                    'ciphertext': ciphertext
                }
                
                if config['mode'] == 'GCM':
                    result['tag'] = encryptor.tag
                
                # Update statistics
                self._update_encryption_stats(container_id, len(data), len(ciphertext))
                
                return result
                
        except Exception as e:
            logger.error(f"Encryption failed: {str(e)}")
            raise

    def decrypt_data(self, container_id: str, encrypted_data: Dict) -> bytes:
        """Decrypt data using stored parameters"""
        try:
            with self.lock:
                if container_id not in self.active_keys:
                    raise ValueError(f"No encryption configuration for container {container_id}")
                
                key_info = self.active_keys[container_id]
                key = key_info['key']
                config = key_info['config']
                
                # Create cipher
                if config['mode'] == 'GCM':
                    cipher = Cipher(
                        algorithms.AES(key),
                        modes.GCM(encrypted_data['iv'], encrypted_data['tag'])
                    )
                else:  # CBC mode
                    cipher = Cipher(
                        algorithms.AES(key),
                        modes.CBC(encrypted_data['iv'])
                    )
                
                decryptor = cipher.decryptor()
                
                # Decrypt data
                padded_data = decryptor.update(encrypted_data['ciphertext'])
                padded_data += decryptor.finalize()
                
                # Remove padding if needed
                if config['mode'] == 'CBC':
                    unpadder = padding.PKCS7(128).unpadder()
                    data = unpadder.update(padded_data) + unpadder.finalize()
                else:
                    data = padded_data
                
                return data
                
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise

    def _adjust_parameters(self, 
                         base_config: Dict, 
                         network_metrics: NetworkMetrics) -> Dict:
        """Adjust encryption parameters based on network conditions"""
        config = base_config.copy()
        
        # Calculate network quality score (0 to 1)
        network_quality = self._calculate_network_quality(network_metrics)
        
        # Adjust parameters based on network quality
        if network_quality < 0.5:
            # Poor network conditions - optimize for performance
            config['iterations'] = max(100000, config['iterations'] // 2)
            if config['memory_hard']:
                config['memory_hard'] = False
        else:
            # Good network conditions - can use stronger parameters
            config['iterations'] = min(300000, config['iterations'] * 1.2)
        
        return config

    def _generate_key(self, config: Dict) -> Tuple[bytes, bytes]:
        """Generate encryption key using key derivation function"""
        salt = os.urandom(16)
        
        # Create key derivation function
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=config['key_size'] // 8,
            salt=salt,
            iterations=config['iterations']
        )
        
        # Generate key
        key = kdf.derive(os.urandom(32))
        
        return key, salt

    def _calculate_network_quality(self, metrics: NetworkMetrics) -> float:
        """Calculate network quality score between 0 and 1"""
        bandwidth_score = min(1.0, metrics.bandwidth / 100)  # Normalize to 100 Mbps
        latency_score = max(0.0, 1.0 - metrics.latency / 200)  # Normalize to 200ms
        packet_loss_score = max(0.0, 1.0 - metrics.packet_loss * 50)
        jitter_score = max(0.0, 1.0 - metrics.jitter / 50)
        
        # Weighted average
        weights = [0.4, 0.3, 0.2, 0.1]
        quality = np.average(
            [bandwidth_score, latency_score, packet_loss_score, jitter_score],
            weights=weights
        )
        
        return float(quality)

    def _update_encryption_stats(self, 
                               container_id: str, 
                               input_size: int, 
                               output_size: int):
        """Update encryption statistics"""
        if container_id not in self.encryption_stats:
            self.encryption_stats[container_id] = {
                'total_bytes': 0,
                'total_time': 0.0,
                'operations': 0
            }
        
        stats = self.encryption_stats[container_id]
        stats['total_bytes'] += input_size
        stats['operations'] += 1

    def _get_public_parameters(self, config: Dict) -> Dict:
        """Get non-sensitive configuration parameters"""
        return {
            'algorithm': config['algorithm'],
            'key_size': config['key_size'],
            'mode': config['mode'],
            'memory_hard': config['memory_hard']
        }

    def get_encryption_stats(self, container_id: str) -> Optional[Dict]:
        """Get encryption statistics for a container"""
        with self.lock:
            return self.encryption_stats.get(container_id)

    def rotate_key(self, container_id: str) -> Dict:
        """Rotate encryption key for a container"""
        try:
            with self.lock:
                if container_id not in self.active_keys:
                    raise ValueError(f"No encryption configuration for container {container_id}")
                
                key_info = self.active_keys[container_id]
                config = key_info['config']
                
                # Generate new key
                new_key, new_salt = self._generate_key(config)
                
                # Update stored key information
                self.active_keys[container_id].update({
                    'key': new_key,
                    'salt': new_salt,
                    'created_at': datetime.now()
                })
                
                return {
                    'status': 'success',
                    'container_id': container_id,
                    'rotated_at': datetime.now()
                }
                
        except Exception as e:
            logger.error(f"Key rotation failed: {str(e)}")
            raise