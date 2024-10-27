import numpy as np
from typing import Dict, List, Tuple
import logging
from ..utils.data_structures import SecurityLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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