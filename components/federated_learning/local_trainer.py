import numpy as np
from typing import Dict, Any
import torch
import torch.nn as nn

class LocalModelTrainer:
    def __init__(self, privacy_budget: float, noise_scale: float):
        self.privacy_budget = privacy_budget
        self.noise_scale = noise_scale
        self.privacy_accountant = PrivacyAccountant()
        
    def train(self, model: nn.Module, data: torch.Tensor, 
              learning_rate: float) -> Dict[str, Any]:
        gradients = []
        
        while self.privacy_accountant.get_budget() < self.privacy_budget:
            # Sample batch with replacement
            batch = self._sample_with_replacement(data)
            
            # Compute gradients
            batch_gradients = self._compute_gradients(model, batch)
            
            # Clip gradients
            clipped_grads = self._clip_gradients(batch_gradients)
            
            # Add Gaussian noise for differential privacy
            noisy_grads = self._add_gaussian_noise(clipped_grads)
            
            # Update model
            self._update_model(model, noisy_grads, learning_rate)
            
            # Update privacy budget
            self.privacy_accountant.update()
            
            gradients.append(noisy_grads)
            
        return {
            'updated_model': model,
            'gradients': gradients,
            'privacy_cost': self.privacy_accountant.get_budget()
        }
    
    def _clip_gradients(self, gradients: torch.Tensor) -> torch.Tensor:
        """Implement gradient clipping as per paper's specifications"""
        norm = torch.norm(gradients)
        scale = min(1.0, self.clip_threshold / norm)
        return gradients * scale
    
    def _add_gaussian_noise(self, gradients: torch.Tensor) -> torch.Tensor:
        """Add calibrated Gaussian noise for differential privacy"""
        noise = torch.randn_like(gradients) * self.noise_scale
        return gradients + noise
