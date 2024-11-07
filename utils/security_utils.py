
import os
import hashlib
import hmac
import base64
from typing import Dict, Any, Tuple, Optional
import logging
from datetime import datetime
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.fernet import Fernet

class KeyManager:
    """Utility class for key management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize key management components"""
        self.key_cache = {}
        self.key_rotation_interval = self.config.get('key_rotation_interval', 86400)
        self.last_rotation = {}

    def generate_key_pair(self, key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate asymmetric key pair"""
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            public_key = private_key.public_key()
            
            return private_key, public_key
            
        except Exception as e:
            self.logger.error(f"Key pair generation failed: {str(e)}")
            raise

    def derive_key(self, 
                  master_key: bytes,
                  salt: bytes,
                  key_size: int = 256) -> bytes:
        """Derive encryption key"""
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_size // 8,
                salt=salt,
                iterations=self.config.get('kdf_iterations', 100000)
            )
            
            return kdf.derive(master_key)
            
        except Exception as e:
            self.logger.error(f"Key derivation failed: {str(e)}")
            raise

    def rotate_keys(self):
        """Rotate encryption keys"""
        try:
            current_time = datetime.now()
            
            for key_id, key_data in self.key_cache.items():
                if (current_time - key_data['created_at']).total_seconds() > \
                   self.key_rotation_interval:
                    # Generate new key
                    new_key = self._generate_key()
                    
                    # Update cache
                    self.key_cache[key_id] = {
                        'key': new_key,
                        'created_at': current_time
                    }
                    
                    self.last_rotation[key_id] = current_time
                    
        except Exception as e:
            self.logger.error(f"Key rotation failed: {str(e)}")

    def _generate_key(self) -> bytes:
        """Generate symmetric key"""
        return os.urandom(32)

class SecurityValidator:
    """Utility class for security validation"""
    
    @staticmethod
    def validate_hash(data: bytes, hash_value: str) -> bool:
        """Validate hash value"""
        computed_hash = hashlib.sha256(data).hexdigest()
        return hmac.compare_digest(computed_hash, hash_value)

    @staticmethod
    def validate_signature(data: bytes,
                         signature: bytes,
                         public_key: Any) -> bool:
        """Validate digital signature"""
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    @staticmethod
    def validate_mac(data: bytes,
                    mac: bytes,
                    key: bytes) -> bool:
        """Validate message authentication code"""
        computed_mac = hmac.new(key, data, hashlib.sha256).digest()
        return hmac.compare_digest(computed_mac, mac)

class SecurityAuditor:
    """Utility class for security auditing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.audit_log = []

    def log_security_event(self,
                         event_type: str,
                         details: Dict[str, Any]):
        """Log security event"""
        event = {
            'timestamp': datetime.now(),
            'type': event_type,
            'details': details
        }
        self.audit_log.append(event)
        self._write_to_log(event)

    def _write_to_log(self, event: Dict[str, Any]):
        """Write event to audit log"""
        try:
            log_file = self.config.get('audit_log_file', 'security_audit.log')
            with open(log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - "
                       f"{event['type']}: {str(event['details'])}\n")
        except Exception as e:
            self.logger.error(f"Audit logging failed: {str(e)}")

class EncryptionUtils:
    """Utility class for encryption operations"""
    
    @staticmethod
    def encrypt_data(data: bytes,
                    key: bytes,
                    algorithm: str = 'AES-GCM') -> Tuple[bytes, bytes, bytes]:
        """Encrypt data"""
        # Generate nonce/IV
        nonce = os.urandom(12)
        
        if algorithm == 'AES-GCM':
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce)
            )
            encryptor = cipher.encryptor()
            
            # Encrypt data
            ciphertext = encryptor.update(data) + encryptor.finalize()
            
            return ciphertext, nonce, encryptor.tag
            
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    @staticmethod
    def decrypt_data(ciphertext: bytes,
                    key: bytes,
                    nonce: bytes,
                    tag: bytes,
                    algorithm: str = 'AES-GCM') -> bytes:
        """Decrypt data"""
        if algorithm == 'AES-GCM':
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(nonce, tag)
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            return decryptor.update(ciphertext) + decryptor.finalize()
            
        raise ValueError(f"Unsupported algorithm: {algorithm}")

class SecurityUtils:
    """Main security utilities class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.key_manager = KeyManager(config.get('key_manager', {}))
        self.auditor = SecurityAuditor(config.get('auditor', {}))
        self.logger = logging.getLogger(__name__)

    def generate_secure_token(self, length: int = 32) -> str:
        """Generate secure token"""
        return base64.urlsafe_b64encode(os.urandom(length)).decode('utf-8')

    def hash_password(self, password: str) -> Tuple[str, str]:
        """Hash password with salt"""
        salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt,
            100000
        )
        return base64.b64encode(key).decode('utf-8'), \
               base64.b64encode(salt).decode('utf-8')

    def verify_password(self,
                       password: str,
                       hash_value: str,
                       salt: str) -> bool:
        """Verify password hash"""
        salt_bytes = base64.b64decode(salt)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode(),
            salt_bytes,
            100000
        )
        return base64.b64encode(key).decode('utf-8') == hash_value

    def encrypt_sensitive_data(self,
                             data: Dict[str, Any]) -> Tuple[bytes, bytes]:
        """Encrypt sensitive data"""
        try:
            # Generate key
            key = Fernet.generate_key()
            f = Fernet(key)
            
            # Encrypt data
            encrypted_data = f.encrypt(
                bytes(str(data), 'utf-8')
            )
            
            return encrypted_data, key
            
        except Exception as e:
            self.logger.error(f"Data encryption failed: {str(e)}")
            raise

    def decrypt_sensitive_data(self,
                             encrypted_data: bytes,
                             key: bytes) -> Dict[str, Any]:
        """Decrypt sensitive data"""
        try:
            f = Fernet(key)
            decrypted_data = f.decrypt(encrypted_data)
            return eval(decrypted_data.decode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Data decryption failed: {str(e)}")
            raise