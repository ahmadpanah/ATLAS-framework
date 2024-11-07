
import torch
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms"""
    AES_256_GCM = "AES-256-GCM"
    CHACHA20_POLY1305 = "CHACHA20-POLY1305"
    AES_256_CBC = "AES-256-CBC"
    AES_128_GCM = "AES-128-GCM"

@dataclass
class AlgorithmSelection:
    """Algorithm selection result"""
    algorithm: EncryptionAlgorithm
    confidence: float
    parameters: Dict[str, Any]
    estimated_performance: float
    timestamp: datetime

class LinUCBAlgorithmSelector:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Algorithm Selector using LinUCB
        
        Args:
            config: Configuration dictionary containing:
                - alpha: Exploration parameter
                - context_dim: Context vector dimension
                - algorithms: List of available algorithms
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize selector components"""
        self.alpha = self.config.get('alpha', 0.5)
        self.context_dim = self.config.get('context_dim', 10)
        
        # Initialize available algorithms
        self.algorithms = [algo for algo in EncryptionAlgorithm]
        
        # Initialize LinUCB parameters for each algorithm
        self.algorithm_params = {}
        for algorithm in self.algorithms:
            self.algorithm_params[algorithm] = {
                'A': np.identity(self.context_dim),  # coefficient matrix
                'b': np.zeros((self.context_dim, 1)),  # offset vector
                'theta': np.zeros((self.context_dim, 1))  # parameter vector
            }

    def select_algorithm(self, context: np.ndarray) -> AlgorithmSelection:
        """
        Select encryption algorithm using LinUCB
        
        Args:
            context: Context vector containing current conditions
            
        Returns:
            AlgorithmSelection object containing selected algorithm
        """
        try:
            # Reshape context vector
            x = context.reshape(-1, 1)
            
            # Calculate UCB for each algorithm
            ucb_scores = {}
            for algorithm in self.algorithms:
                params = self.algorithm_params[algorithm]
                
                # Calculate theta
                A_inv = np.linalg.inv(params['A'])
                params['theta'] = A_inv.dot(params['b'])
                
                # Calculate UCB score
                pred = params['theta'].T.dot(x)
                var = np.sqrt(x.T.dot(A_inv).dot(x))
                ucb = pred + self.alpha * var
                
                ucb_scores[algorithm] = (float(ucb), float(var))
            
            # Select algorithm with highest UCB score
            selected_algorithm = max(ucb_scores.items(), 
                                  key=lambda x: x[1][0])[0]
            confidence = 1.0 / (1.0 + ucb_scores[selected_algorithm][1])
            
            # Get parameters for selected algorithm
            parameters = self._get_algorithm_parameters(selected_algorithm)
            
            # Estimate performance
            estimated_performance = float(
                self.algorithm_params[selected_algorithm]['theta'].T.dot(x)
            )
            
            return AlgorithmSelection(
                algorithm=selected_algorithm,
                confidence=confidence,
                parameters=parameters,
                estimated_performance=estimated_performance,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Algorithm selection failed: {str(e)}")
            # Fallback to default algorithm
            return self._get_default_selection()

    def update_model(self, 
                    algorithm: EncryptionAlgorithm,
                    context: np.ndarray,
                    reward: float):
        """Update model with observed reward"""
        try:
            # Reshape context vector
            x = context.reshape(-1, 1)
            
            # Update algorithm parameters
            params = self.algorithm_params[algorithm]
            params['A'] += x.dot(x.T)
            params['b'] += reward * x
            
        except Exception as e:
            self.logger.error(f"Model update failed: {str(e)}")

    def _get_algorithm_parameters(self, 
                                algorithm: EncryptionAlgorithm) -> Dict[str, Any]:
        """Get default parameters for algorithm"""
        parameters = {
            EncryptionAlgorithm.AES_256_GCM: {
                'key_size': 256,
                'mode': 'GCM',
                'nonce_size': 12,
                'tag_size': 16
            },
            EncryptionAlgorithm.CHACHA20_POLY1305: {
                'key_size': 256,
                'nonce_size': 12,
                'tag_size': 16
            },
            EncryptionAlgorithm.AES_256_CBC: {
                'key_size': 256,
                'mode': 'CBC',
                'iv_size': 16
            },
            EncryptionAlgorithm.AES_128_GCM: {
                'key_size': 128,
                'mode': 'GCM',
                'nonce_size': 12,
                'tag_size': 16
            }
        }
        return parameters.get(algorithm, {})

    def _get_default_selection(self) -> AlgorithmSelection:
        """Get default algorithm selection"""
        return AlgorithmSelection(
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            confidence=1.0,
            parameters=self._get_algorithm_parameters(
                EncryptionAlgorithm.AES_256_GCM
            ),
            estimated_performance=0.0,
            timestamp=datetime.now()
        )

    def get_algorithm_statistics(self) -> Dict[str, Any]:
        """Get algorithm selection statistics"""
        stats = {}
        for algorithm in self.algorithms:
            params = self.algorithm_params[algorithm]
            stats[algorithm.value] = {
                'theta': params['theta'].flatten().tolist(),
                'uncertainty': np.linalg.det(np.linalg.inv(params['A']))
            }
        return stats