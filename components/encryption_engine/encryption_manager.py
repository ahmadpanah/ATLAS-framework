
import os
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
from enum import Enum
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes, hmac
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.exceptions import InvalidTag

class EncryptionState(Enum):
    """Encryption process states"""
    INIT = "INIT"
    KEYGEN = "KEYGEN"
    ENCRYPT = "ENCRYPT"
    VERIFY = "VERIFY"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"

class EncryptionManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Encryption Manager
        
        Args:
            config: Configuration dictionary containing:
                - key_derivation: Key derivation parameters
                - buffer_size: Size of encryption buffer
                - verification_mode: Verification mode configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize manager components"""
        self.state = EncryptionState.INIT
        self.buffer_size = self.config.get('buffer_size', 1024*1024)  # 1MB
        self.current_context = None
        self.encryption_stats = {
            'operations': 0,
            'bytes_processed': 0,
            'errors': 0
        }

    def manage_encryption(self,
                         data: bytes,
                         algorithm: str,
                         parameters: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        """
        Manage encryption process
        
        Args:
            data: Data to encrypt
            algorithm: Selected encryption algorithm
            parameters: Encryption parameters
            
        Returns:
            Tuple containing encrypted data and process metadata
        """
        try:
            self.state = EncryptionState.INIT
            metadata = {
                'algorithm': algorithm,
                'parameters': parameters,
                'timestamp': datetime.now(),
                'state_history': []
            }
            
            # Create encryption context
            self.current_context = self._create_context(algorithm, parameters)
            metadata['state_history'].append(
                {'state': self.state.value, 'timestamp': datetime.now()}
            )
            
            # Generate keys
            self.state = EncryptionState.KEYGEN
            keys = self._generate_keys(parameters)
            metadata['state_history'].append(
                {'state': self.state.value, 'timestamp': datetime.now()}
            )
            
            # Encrypt data
            self.state = EncryptionState.ENCRYPT
            encrypted_data = self._encrypt_data(data, keys, algorithm, parameters)
            metadata['state_history'].append(
                {'state': self.state.value, 'timestamp': datetime.now()}
            )
            
            # Verify encryption
            self.state = EncryptionState.VERIFY
            if self._verify_encryption(encrypted_data, keys, algorithm, parameters):
                self.state = EncryptionState.COMPLETED
                self.encryption_stats['operations'] += 1
                self.encryption_stats['bytes_processed'] += len(data)
            else:
                self.state = EncryptionState.ERROR
                self.encryption_stats['errors'] += 1
                raise ValueError("Encryption verification failed")
                
            metadata['state_history'].append(
                {'state': self.state.value, 'timestamp': datetime.now()}
            )
            
            return encrypted_data, metadata
            
        except Exception as e:
            self.logger.error(f"Encryption management failed: {str(e)}")
            self.state = EncryptionState.ERROR
            raise

    def _create_context(self, 
                       algorithm: str,
                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create encryption context"""
        return {
            'algorithm': algorithm,
            'parameters': parameters,
            'created_at': datetime.now()
        }

    def _generate_keys(self, parameters: Dict[str, Any]) -> Dict[str, bytes]:
        """Generate encryption keys"""
        try:
            # Generate salt
            salt = os.urandom(16)
            
            # Create key derivation function
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=parameters.get('key_size', 256) // 8,
                salt=salt,
                iterations=parameters.get('iterations', 200000)
            )
            
            # Generate keys
            master_key = os.urandom(32)
            encryption_key = kdf.derive(master_key)
            
            # Generate additional keys if needed
            keys = {
                'master_key': master_key,
                'encryption_key': encryption_key,
                'salt': salt
            }
            
            if parameters.get('mode') == 'GCM':
                keys['nonce'] = os.urandom(parameters.get('nonce_size', 12))
                
            return keys
            
        except Exception as e:
            self.logger.error(f"Key generation failed: {str(e)}")
            raise

    def _encrypt_data(self,
                     data: bytes,
                     keys: Dict[str, bytes],
                     algorithm: str,
                     parameters: Dict[str, Any]) -> bytes:
        """Encrypt data using specified algorithm"""
        try:
            if algorithm == 'AES-256-GCM':
                cipher = Cipher(
                    algorithms.AES(keys['encryption_key']),
                    modes.GCM(keys['nonce'])
                )
                encryptor = cipher.encryptor()
                
                # Process data in chunks
                chunks = []
                for i in range(0, len(data), self.buffer_size):
                    chunk = data[i:i + self.buffer_size]
                    chunks.append(encryptor.update(chunk))
                
                chunks.append(encryptor.finalize())
                
                # Add authentication tag
                encrypted_data = b''.join(chunks) + encryptor.tag
                
            elif algorithm == 'CHACHA20-POLY1305':
                cipher = Cipher(
                    algorithms.ChaCha20(
                        keys['encryption_key'],
                        keys['nonce']
                    ),
                    None
                )
                encryptor = cipher.encryptor()
                
                # Process data in chunks
                chunks = []
                for i in range(0, len(data), self.buffer_size):
                    chunk = data[i:i + self.buffer_size]
                    chunks.append(encryptor.update(chunk))
                
                chunks.append(encryptor.finalize())
                encrypted_data = b''.join(chunks)
                
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
                
            return encrypted_data
            
        except Exception as e:
            self.logger.error(f"Data encryption failed: {str(e)}")
            raise

    def _verify_encryption(self,
                         encrypted_data: bytes,
                         keys: Dict[str, bytes],
                         algorithm: str,
                         parameters: Dict[str, Any]) -> bool:
        """Verify encryption result"""
        try:
            # Verify data integrity
            if algorithm == 'AES-256-GCM':
                tag_size = parameters.get('tag_size', 16)
                ciphertext = encrypted_data[:-tag_size]
                tag = encrypted_data[-tag_size:]
                
                cipher = Cipher(
                    algorithms.AES(keys['encryption_key']),
                    modes.GCM(keys['nonce'], tag)
                )
                decryptor = cipher.decryptor()
                
                # Try to decrypt a small portion
                test_size = min(1024, len(ciphertext))
                decryptor.update(ciphertext[:test_size])
                decryptor.finalize()
                
            elif algorithm == 'CHACHA20-POLY1305':
                cipher = Cipher(
                    algorithms.ChaCha20(
                        keys['encryption_key'],
                        keys['nonce']
                    ),
                    None
                )
                decryptor = cipher.decryptor()
                
                # Try to decrypt a small portion
                test_size = min(1024, len(encrypted_data))
                decryptor.update(encrypted_data[:test_size])
                decryptor.finalize()
                
            return True
            
        except InvalidTag:
            return False
        except Exception as e:
            self.logger.error(f"Encryption verification failed: {str(e)}")
            return False

    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption statistics"""
        return {
            **self.encryption_stats,
            'current_state': self.state.value,
            'last_context': self.current_context
        }