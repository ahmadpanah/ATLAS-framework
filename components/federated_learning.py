import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import os
import logging
from ..utils.data_structures import SecurityLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 10
    privacy_budget: float = 1.0
    noise_scale: float = 0.01
    min_clients: int = 3
    max_clients: int = 10
    aggregation_rounds: int = 5

class LocalModelTrainer:
    """Implementation of Algorithm 1: Local Model Training"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.privacy_engine = None
        self.local_dataset = None
        
    def initialize_model(self, model_architecture: nn.Module):
        """Initialize the local model with given architecture"""
        self.model = model_architecture
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )

    def add_differential_privacy(self):
        """Add differential privacy noise to gradients"""
        noise_multiplier = self.config.noise_scale
        max_grad_norm = 1.0
        
        def add_noise_to_gradients(param):
            if param.grad is not None:
                noise = torch.normal(
                    mean=0,
                    std=noise_multiplier * max_grad_norm,
                    size=param.grad.shape
                )
                param.grad += noise

        return add_noise_to_gradients

    def train_local_model(self, 
                         local_dataset: torch.utils.data.Dataset) -> Tuple[Dict, Dict]:
        """
        Implementation of Algorithm 1: Local Model Training
        Returns: (model_parameters, training_metrics)
        """
        try:
            logger.info("Starting local model training...")
            self.local_dataset = local_dataset
            dataloader = torch.utils.data.DataLoader(
                local_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            dp_hook = self.add_differential_privacy()
            training_losses = []
            privacy_spent = 0.0
            
            # Main training loop as per Algorithm 1
            for epoch in range(self.config.epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for batch_data, batch_labels in dataloader:
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(batch_data)
                    loss = nn.functional.cross_entropy(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Add differential privacy noise to gradients
                    for param in self.model.parameters():
                        dp_hook(param)
                    
                    # Update weights
                    self.optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Update privacy budget spent
                    privacy_spent += self.config.noise_scale
                    
                    # Check privacy budget
                    if privacy_spent >= self.config.privacy_budget:
                        logger.warning("Privacy budget exceeded, stopping training")
                        break
                
                avg_epoch_loss = epoch_loss / batch_count
                training_losses.append(avg_epoch_loss)
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs}, "
                          f"Loss: {avg_epoch_loss:.4f}, "
                          f"Privacy spent: {privacy_spent:.4f}")
            
            # Prepare results
            model_parameters = {
                name: param.data.clone()
                for name, param in self.model.state_dict().items()
            }
            
            training_metrics = {
                'losses': training_losses,
                'privacy_spent': privacy_spent,
                'completed_epochs': len(training_losses),
                'final_loss': training_losses[-1]
            }
            
            return model_parameters, training_metrics
            
        except Exception as e:
            logger.error(f"Local training failed: {str(e)}")
            raise

class SecureAggregator:
    """Implementation of Algorithm 2: Secure Model Aggregation"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.encryption_keys = {}
        self.masks = {}
        self.masked_models = {}
        self.initialize_security()
        
    def initialize_security(self):
        """Initialize security components"""
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        self.symmetric_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.symmetric_key)
        
    def generate_mask(self, model_shape: Dict) -> Dict:
        """Generate random mask for model parameters"""
        mask = {}
        for name, shape in model_shape.items():
            mask[name] = torch.randn_like(torch.zeros(shape))
        return mask
        
    def secure_aggregate(self, 
                        local_models: List[Dict],
                        security_threshold: float) -> Dict:
        """
        Implementation of Algorithm 2: Secure Model Aggregation
        Returns: aggregated global model
        """
        try:
            logger.info("Starting secure model aggregation...")
            
            if len(local_models) < self.config.min_clients:
                raise ValueError(
                    f"Insufficient clients for aggregation. "
                    f"Minimum required: {self.config.min_clients}"
                )
            
            # Step 1: Initialize secure communication
            client_keys = self._setup_secure_channels(len(local_models))
            
            # Step 2: Generate and distribute masks
            for client_id, model in enumerate(local_models):
                mask = self.generate_mask(
                    {name: param.shape for name, param in model.items()}
                )
                self.masks[client_id] = mask
                
                # Encrypt and store mask
                encrypted_mask = self._encrypt_mask(mask, client_keys[client_id])
                self._distribute_mask(client_id, encrypted_mask)
            
            # Step 3: Apply masks to local models
            for client_id, model in enumerate(local_models):
                masked_model = self._apply_mask(model, self.masks[client_id])
                self.masked_models[client_id] = masked_model
            
            # Step 4: Aggregate masks
            aggregate_mask = self._aggregate_masks(list(self.masks.values()))
            
            # Step 5: Aggregate masked models
            aggregated_model = {}
            num_models = len(local_models)
            
            for name in local_models[0].keys():
                # Sum all masked parameters
                param_sum = sum(
                    self.masked_models[client_id][name] 
                    for client_id in range(num_models)
                )
                
                # Remove aggregate mask and average
                aggregated_model[name] = (param_sum - aggregate_mask[name]) / num_models
            
            # Verify aggregation security
            if not self._verify_security(aggregated_model, security_threshold):
                raise SecurityError("Security verification failed during aggregation")
            
            logger.info("Secure model aggregation completed successfully")
            return aggregated_model
            
        except Exception as e:
            logger.error(f"Secure aggregation failed: {str(e)}")
            raise
            
    def _setup_secure_channels(self, num_clients: int) -> Dict:
        """Setup secure communication channels with clients"""
        client_keys = {}
        for client_id in range(num_clients):
            # Generate unique key for each client
            client_key = Fernet.generate_key()
            client_keys[client_id] = client_key
            
            # Encrypt client key with public key
            encrypted_key = self.public_key.encrypt(
                client_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            self.encryption_keys[client_id] = encrypted_key
            
        return client_keys
        
    def _encrypt_mask(self, mask: Dict, key: bytes) -> Dict:
        """Encrypt mask with client-specific key"""
        encrypted_mask = {}
        cipher_suite = Fernet(key)
        
        for name, param in mask.items():
            # Convert tensor to bytes
            param_bytes = param.numpy().tobytes()
            # Encrypt bytes
            encrypted_param = cipher_suite.encrypt(param_bytes)
            encrypted_mask[name] = encrypted_param
            
        return encrypted_mask
        
    def _distribute_mask(self, client_id: int, encrypted_mask: Dict):
        """Distribute encrypted mask to client"""
        # In real implementation, this would involve network communication
        # Here we just store it locally
        self.masks[client_id] = encrypted_mask
        
    def _apply_mask(self, model: Dict, mask: Dict) -> Dict:
        """Apply mask to model parameters"""
        masked_model = {}
        for name, param in model.items():
            masked_model[name] = param + mask[name]
        return masked_model
        
    def _aggregate_masks(self, masks: List[Dict]) -> Dict:
        """Aggregate all masks"""
        aggregate_mask = {}
        for name in masks[0].keys():
            aggregate_mask[name] = sum(mask[name] for mask in masks)
        return aggregate_mask
        
    def _verify_security(self, aggregated_model: Dict, threshold: float) -> bool:
        """Verify security of aggregated model"""
        try:
            # Check for anomalous values
            for name, param in aggregated_model.items():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    return False
                    
                # Check value range
                if param.abs().max() > threshold:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Security verification failed: {str(e)}")
            return False

class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass

def initialize_federated_learning() -> Tuple[LocalModelTrainer, SecureAggregator]:
    """Initialize federated learning components"""
    try:
        config = TrainingConfig()
        trainer = LocalModelTrainer(config)
        aggregator = SecureAggregator(config)
        return trainer, aggregator
    except Exception as e:
        logger.error(f"Failed to initialize federated learning: {str(e)}")
        raise

class FederatedLearningModule:
    def __init__(self):
        self.local_models: Dict[str, np.ndarray] = {}
        self.global_model: Optional[np.ndarray] = None
        self.learning_rate = 0.01
        self.privacy_noise_scale = 0.01
        self.threat_patterns = {}

    def train_local_model(self, provider_id: str, threat_data: List[float]) -> Dict:
        """
        Train a local model with differential privacy
        Implements Algorithm 1 from the paper
        """
        try:
            # Convert data to numpy array
            data = np.array(threat_data)
            
            # Add differential privacy noise
            noise = np.random.normal(0, self.privacy_noise_scale, data.shape)
            private_data = data + noise
            
            # Update local model using gradient descent
            if provider_id in self.local_models:
                current_model = self.local_models[provider_id]
                gradient = self._compute_gradient(current_model, private_data)
                updated_model = current_model - self.learning_rate * gradient
            else:
                updated_model = private_data
            
            # Store updated model
            self.local_models[provider_id] = updated_model
            
            return {
                "status": "success",
                "provider_id": provider_id,
                "model_size": len(updated_model)
            }
            
        except Exception as e:
            logger.error(f"Local model training failed: {str(e)}")
            raise

    def aggregate_global_model(self) -> Dict:
        """
        Aggregate local models using secure aggregation
        Implements Algorithm 2 from the paper
        """
        try:
            if not self.local_models:
                raise ValueError("No local models available for aggregation")

            # Generate random masks for secure aggregation
            masks = self._generate_masks()
            
            # Apply masks to local models
            masked_models = {}
            for provider_id, model in self.local_models.items():
                masked_models[provider_id] = model + masks[provider_id]
            
            # Aggregate masked models
            aggregate = np.mean(list(masked_models.values()), axis=0)
            
            # Remove masks from aggregate
            total_mask = sum(masks.values())
            self.global_model = aggregate - (total_mask / len(self.local_models))
            
            # Update threat patterns
            self._update_threat_patterns()
            
            return {
                "status": "success",
                "participating_providers": len(self.local_models),
                "global_model_size": len(self.global_model)
            }
            
        except Exception as e:
            logger.error(f"Global model aggregation failed: {str(e)}")
            raise

    def _compute_gradient(self, model: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Compute gradient for model update"""
        # Simplified gradient computation
        return model - data

    def _generate_masks(self) -> Dict[str, np.ndarray]:
        """Generate random masks for secure aggregation"""
        masks = {}
        for provider_id, model in self.local_models.items():
            masks[provider_id] = np.random.normal(0, 0.1, model.shape)
        return masks

    def _update_threat_patterns(self):
        """Update known threat patterns based on global model"""
        if self.global_model is not None:
            # Extract significant patterns
            threshold = np.mean(self.global_model) + np.std(self.global_model)
            significant_patterns = self.global_model > threshold
            
            # Update threat patterns dictionary
            for i, is_significant in enumerate(significant_patterns):
                if is_significant:
                    self.threat_patterns[f"pattern_{i}"] = float(self.global_model[i])

    def assess_security_requirements(self, container_attributes: Dict) -> Tuple[SecurityLevel, float]:
        """Assess container security requirements based on threat patterns"""
        if not self.threat_patterns:
            return SecurityLevel.MEDIUM, 0.5

        # Calculate risk score based on threat patterns
        risk_scores = []
        for pattern_value in self.threat_patterns.values():
            # Compare container attributes with threat patterns
            risk_score = self._calculate_risk_score(container_attributes, pattern_value)
            risk_scores.append(risk_score)

        # Aggregate risk scores
        final_risk_score = np.mean(risk_scores)

        # Map risk score to security level
        if final_risk_score > 0.8:
            return SecurityLevel.CRITICAL, final_risk_score
        elif final_risk_score > 0.6:
            return SecurityLevel.HIGH, final_risk_score
        elif final_risk_score > 0.4:
            return SecurityLevel.MEDIUM, final_risk_score
        else:
            return SecurityLevel.LOW, final_risk_score

    def _calculate_risk_score(self, container_attributes: Dict, pattern_value: float) -> float:
        """Calculate risk score for a container based on a threat pattern"""
        # Simplified risk calculation
        base_score = pattern_value
        
        # Adjust score based on container attributes
        if container_attributes.get('exposed_ports', []):
            base_score *= 1.2
        if container_attributes.get('volume_mounts', []):
            base_score *= 1.1
        if not container_attributes.get('resource_limits', {}):
            base_score *= 1.3
            
        return min(1.0, base_score)