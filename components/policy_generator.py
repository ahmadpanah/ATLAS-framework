# components/policy_generator.py

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from enum import Enum
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class FeatureSet:
    model_weights: np.ndarray
    layer_statistics: Dict[str, np.ndarray]
    threat_indicators: Dict[str, float]
    network_metrics: Dict[str, float]
    container_metrics: Dict[str, Any]

@dataclass
class EncryptionPolicy:
    algorithm: str
    key_size: int
    mode: str
    padding: str
    iterations: int
    memory_cost: int
    parallelism: int
    additional_params: Dict[str, Any]
    timestamp: datetime
    expiry: datetime
    policy_id: str

class PolicyGenerator:
    """Implementation of Algorithm 3: Encryption Policy Generation"""
    
    def __init__(self):
        self.feature_extractor = LayerFeatureExtractor()
        self.risk_assessor = RiskAssessor()
        self.policy_validator = PolicyValidator()
        
    def generate_policy(self, 
                       global_model: Dict[str, torch.Tensor],
                       threat_landscape: Dict[str, Any]) -> EncryptionPolicy:
        """
        Implementation of Algorithm 3: Main policy generation function
        """
        try:
            logger.info("Starting encryption policy generation...")
            
            # Extract features from global model
            features = self.feature_extractor.extract_features(global_model)
            
            # Assess risk based on features and threat landscape
            risk_score = self.risk_assessor.assess_risk(features, threat_landscape)
            
            # Generate encryption policy based on risk assessment
            policy = self._generate_encryption_policy(risk_score)
            
            # Validate generated policy
            if not self.policy_validator.validate_policy(policy):
                raise ValueError("Generated policy validation failed")
            
            logger.info(f"Generated encryption policy with ID: {policy.policy_id}")
            return policy
            
        except Exception as e:
            logger.error(f"Policy generation failed: {str(e)}")
            raise

class LayerFeatureExtractor:
    """Feature extraction from neural network layers"""
    
    def extract_features(self, model: Dict[str, torch.Tensor]) -> FeatureSet:
        try:
            features = []
            layer_stats = {}
            
            for layer_name, layer_weights in model.items():
                # Calculate layer statistics
                stats = self._compute_layer_statistics(layer_weights)
                layer_stats[layer_name] = stats
                
                # Extract layer-specific features
                layer_features = self._extract_layer_features(layer_weights)
                features.append(layer_features)
            
            # Combine all features
            combined_features = self._combine_features(features)
            
            return FeatureSet(
                model_weights=combined_features,
                layer_statistics=layer_stats,
                threat_indicators=self._compute_threat_indicators(layer_stats),
                network_metrics=self._get_network_metrics(),
                container_metrics=self._get_container_metrics()
            )
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            raise
            
    def _compute_layer_statistics(self, 
                                layer_weights: torch.Tensor) -> np.ndarray:
        """Compute statistical measures for layer weights"""
        with torch.no_grad():
            stats = {
                'mean': layer_weights.mean().item(),
                'std': layer_weights.std().item(),
                'min': layer_weights.min().item(),
                'max': layer_weights.max().item(),
                'l1_norm': layer_weights.abs().sum().item(),
                'l2_norm': layer_weights.norm().item(),
                'sparsity': (layer_weights == 0).float().mean().item()
            }
            return stats

class RiskAssessor:
    """Risk assessment based on features and threat landscape"""
    
    def assess_risk(self, 
                   features: FeatureSet,
                   threat_landscape: Dict[str, Any]) -> float:
        try:
            # Calculate base risk from model features
            base_risk = self._calculate_base_risk(features)
            
            # Adjust risk based on threat landscape
            adjusted_risk = self._adjust_risk(base_risk, threat_landscape)
            
            # Apply risk weighting
            final_risk = self._apply_risk_weights(adjusted_risk, features)
            
            logger.info(f"Calculated risk score: {final_risk:.4f}")
            return final_risk
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {str(e)}")
            raise

class PolicyValidator:
    """Validation of generated encryption policies"""
    
    def validate_policy(self, policy: EncryptionPolicy) -> bool:
        try:
            # Validate algorithm
            if not self._validate_algorithm(policy.algorithm):
                return False
                
            # Validate key size
            if not self._validate_key_size(policy.algorithm, policy.key_size):
                return False
                
            # Validate mode of operation
            if not self._validate_mode(policy.algorithm, policy.mode):
                return False
                
            # Validate additional parameters
            if not self._validate_additional_params(policy.additional_params):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Policy validation failed: {str(e)}")
            return False

# components/container_analyzer.py

class ContainerFeatureExtractor:
    """Implementation of Algorithm 4: Container Feature Extraction"""
    
    def __init__(self):
        self.static_analyzer = StaticAnalyzer()
        self.dynamic_analyzer = DynamicAnalyzer()
        self.metric_collector = MetricCollector()
        
    def extract_features(self, 
                        container_instance: Any,
                        monitoring_window: int) -> Dict[str, Any]:
        """
        Implementation of Algorithm 4: Main feature extraction function
        """
        try:
            logger.info(f"Starting feature extraction for container: {container_instance.id}")
            
            # Extract static features
            static_features = self.static_analyzer.analyze(container_instance)
            
            # Initialize time series buffer
            time_series_buffer = self._initialize_buffer(monitoring_window)
            
            # Collect dynamic metrics
            for t in range(monitoring_window):
                metrics = self.metric_collector.collect_metrics(container_instance)
                processed_metrics = self._process_metrics(metrics)
                time_series_buffer[t] = processed_metrics
                
            # Compute time series features
            dynamic_features = self.dynamic_analyzer.analyze_time_series(time_series_buffer)
            
            # Combine features
            combined_features = {
                'static': static_features,
                'dynamic': dynamic_features,
                'timestamp': datetime.utcnow().isoformat(),
                'container_id': container_instance.id
            }
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Container feature extraction failed: {str(e)}")
            raise

class StaticAnalyzer:
    """Static analysis of container characteristics"""
    
    def analyze(self, container: Any) -> Dict[str, Any]:
        try:
            return {
                'image': self._analyze_image(container.image),
                'config': self._analyze_config(container.config),
                'security': self._analyze_security_config(container.security),
                'volumes': self._analyze_volumes(container.volumes),
                'network': self._analyze_network_config(container.network),
                'resources': self._analyze_resource_limits(container.resources)
            }
        except Exception as e:
            logger.error(f"Static analysis failed: {str(e)}")
            raise

class DynamicAnalyzer:
    """Dynamic analysis of container behavior"""
    
    def analyze_time_series(self, 
                           time_series_buffer: np.ndarray) -> Dict[str, Any]:
        try:
            return {
                'cpu': self._analyze_cpu_patterns(time_series_buffer),
                'memory': self._analyze_memory_patterns(time_series_buffer),
                'network': self._analyze_network_patterns(time_series_buffer),
                'io': self._analyze_io_patterns(time_series_buffer),
                'syscalls': self._analyze_syscalls(time_series_buffer)
            }
        except Exception as e:
            logger.error(f"Dynamic analysis failed: {str(e)}")
            raise

class MetricCollector:
    """Collection of container runtime metrics"""
    
    def collect_metrics(self, container: Any) -> Dict[str, Any]:
        try:
            return {
                'cpu': self._collect_cpu_metrics(container),
                'memory': self._collect_memory_metrics(container),
                'network': self._collect_network_metrics(container),
                'io': self._collect_io_metrics(container),
                'syscalls': self._collect_syscall_metrics(container)
            }
        except Exception as e:
            logger.error(f"Metric collection failed: {str(e)}")
            raise