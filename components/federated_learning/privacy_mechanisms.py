import numpy as np
from typing import Dict, List, Any, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import phe.paillier as paillier
from collections import OrderedDict
import torch
from torch import nn
import logging

class DifferentialPrivacy:
    """Differential Privacy implementation for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = 1.0
        self.noise_multiplier = self._compute_noise_multiplier()
        self.logger = logging.getLogger(__name__)

    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier based on privacy parameters"""
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def add_noise(self, gradient: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to gradients"""
        try:
            noise_scale = self.noise_multiplier * self.sensitivity
            noise = np.random.normal(0, noise_scale, gradient.shape)
            return gradient + noise
        except Exception as e:
            self.logger.error(f"Error adding noise: {str(e)}")
            raise

    def clip_gradients(self, gradients: np.ndarray, 
                      clip_norm: float = 1.0) -> np.ndarray:
        """Clip gradients to limit sensitivity"""
        try:
            norm = np.linalg.norm(gradients)
            if norm > clip_norm:
                gradients = gradients * (clip_norm / norm)
            return gradients
        except Exception as e:
            self.logger.error(f"Error clipping gradients: {str(e)}")
            raise

class SecureMultiPartyComputation:
    """Secure Multi-Party Computation implementation"""
    
    def __init__(self, num_parties: int, threshold: int):
        self.num_parties = num_parties
        self.threshold = threshold
        self.keys = self._generate_keys()
        self.logger = logging.getLogger(__name__)

    def _generate_keys(self) -> Dict[str, Any]:
        """Generate keys for secure computation"""
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            return {
                'private': private_key,
                'public': public_key
            }
        except Exception as e:
            self.logger.error(f"Error generating keys: {str(e)}")
            raise

    def generate_shares(self, secret: np.ndarray) -> List[np.ndarray]:
        """Generate Shamir's secret shares"""
        try:
            shares = []
            coefficients = [secret] + [np.random.rand(*secret.shape) 
                          for _ in range(self.threshold - 1)]
            
            for i in range(1, self.num_parties + 1):
                share = np.zeros_like(secret)
                for j, coef in enumerate(coefficients):
                    share += coef * (i ** j)
                shares.append(share)
            
            return shares
        except Exception as e:
            self.logger.error(f"Error generating shares: {str(e)}")
            raise

    def reconstruct_secret(self, shares: List[np.ndarray], 
                          indices: List[int]) -> np.ndarray:
        """Reconstruct secret from shares using Lagrange interpolation"""
        try:
            if len(shares) < self.threshold:
                raise ValueError("Not enough shares for reconstruction")
                
            secret = np.zeros_like(shares[0])
            for i, share in enumerate(shares[:self.threshold]):
                basis = 1.0
                for j in range(self.threshold):
                    if i != j:
                        basis *= (indices[j] / (indices[i] - indices[j]))
                secret += share * basis
                
            return secret
        except Exception as e:
            self.logger.error(f"Error reconstructing secret: {str(e)}")
            raise

class HomomorphicEncryption:
    """Homomorphic Encryption implementation"""
    
    def __init__(self, key_length: int = 2048):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(
            n_length=key_length
        )
        self.logger = logging.getLogger(__name__)

    def encrypt_vector(self, vector: np.ndarray) -> List[Any]:
        """Encrypt vector using homomorphic encryption"""
        try:
            return [self.public_key.encrypt(float(x)) for x in vector.flatten()]
        except Exception as e:
            self.logger.error(f"Error encrypting vector: {str(e)}")
            raise

    def decrypt_vector(self, encrypted_vector: List[Any]) -> np.ndarray:
        """Decrypt vector using homomorphic encryption"""
        try:
            decrypted = [self.private_key.decrypt(x) for x in encrypted_vector]
            return np.array(decrypted)
        except Exception as e:
            self.logger.error(f"Error decrypting vector: {str(e)}")
            raise

    def aggregate_encrypted_vectors(self, 
                                  encrypted_vectors: List[List[Any]]) -> List[Any]:
        """Aggregate encrypted vectors"""
        try:
            if not encrypted_vectors:
                return []
                
            result = encrypted_vectors[0]
            for vector in encrypted_vectors[1:]:
                result = [a + b for a, b in zip(result, vector)]
            return result
        except Exception as e:
            self.logger.error(f"Error aggregating vectors: {str(e)}")
            raise

class PrivacyPreservingFederatedLearning:
    """Privacy-Preserving Federated Learning implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.differential_privacy = DifferentialPrivacy(
            epsilon=config.get('epsilon', 1.0),
            delta=config.get('delta', 1e-5)
        )
        self.smpc = SecureMultiPartyComputation(
            num_parties=config.get('num_parties', 10),
            threshold=config.get('threshold', 6)
        )
        self.homomorphic = HomomorphicEncryption(
            key_length=config.get('key_length', 2048)
        )
        self.logger = logging.getLogger(__name__)

    async def train_with_privacy(self, 
                               model: nn.Module,
                               local_data: torch.Tensor,
                               local_labels: torch.Tensor) -> Dict[str, Any]:
        """Perform privacy-preserving federated learning"""
        try:
            # Local training
            gradients = self._compute_gradients(model, local_data, local_labels)
            
            # Apply differential privacy
            clipped_gradients = self.differential_privacy.clip_gradients(gradients)
            noisy_gradients = self.differential_privacy.add_noise(clipped_gradients)
            
            # Generate secure shares
            gradient_shares = self.smpc.generate_shares(noisy_gradients)
            
            # Encrypt shares
            encrypted_shares = [
                self.homomorphic.encrypt_vector(share) 
                for share in gradient_shares
            ]
            
            return {
                'encrypted_shares': encrypted_shares,
                'privacy_budget': self.differential_privacy.epsilon
            }
            
        except Exception as e:
            self.logger.error(f"Error in privacy-preserving training: {str(e)}")
            raise

    def aggregate_with_privacy(self, 
                             encrypted_updates: List[Dict[str, Any]]) -> np.ndarray:
        """Aggregate encrypted model updates"""
        try:
            # Collect encrypted shares
            all_shares = [update['encrypted_shares'] for update in encrypted_updates]
            
            # Aggregate encrypted shares
            aggregated_shares = []
            for i in range(len(all_shares[0])):
                shares_i = [shares[i] for shares in all_shares]
                aggregated_share = self.homomorphic.aggregate_encrypted_vectors(
                    shares_i
                )
                aggregated_shares.append(aggregated_share)
            
            # Decrypt and reconstruct
            decrypted_shares = [
                self.homomorphic.decrypt_vector(share) 
                for share in aggregated_shares
            ]
            
            indices = list(range(1, len(decrypted_shares) + 1))
            final_update = self.smpc.reconstruct_secret(
                decrypted_shares, 
                indices
            )
            
            return final_update
            
        except Exception as e:
            self.logger.error(f"Error in private aggregation: {str(e)}")
            raise

    def _compute_gradients(self, 
                          model: nn.Module,
                          data: torch.Tensor,
                          labels: torch.Tensor) -> np.ndarray:
        """Compute gradients from local training"""
        try:
            model.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, labels)
            loss.backward()
            
            gradients = []
            for param in model.parameters():
                gradients.append(param.grad.numpy())
                
            return np.concatenate([g.flatten() for g in gradients])
            
        except Exception as e:
            self.logger.error(f"Error computing gradients: {str(e)}")
            raise

    def verify_privacy_guarantees(self, 
                                training_history: List[Dict[str, Any]]) -> bool:
        """Verify privacy guarantees are maintained"""
        try:
            total_privacy_cost = sum(
                update['privacy_budget'] 
                for update in training_history
            )
            return total_privacy_cost <= self.config.get('max_privacy_budget', 10.0)
            
        except Exception as e:
            self.logger.error(f"Error verifying privacy: {str(e)}")
            raise