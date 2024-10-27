# components/security_classifier.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from datetime import datetime
from enum import Enum
import json
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1

@dataclass
class SecurityClassification:
    level: SecurityLevel
    confidence: float
    timestamp: datetime
    features: Dict[str, Any]
    risk_factors: Dict[str, float]
    recommendations: List[str]

class ContainerSecurityClassifier:
    """Implementation of Algorithm 5: Container Security Classification"""

    def __init__(self, model_config: Dict[str, Any]):
        self.transformer = TransformerClassifier(model_config)
        self.layer_norm = nn.LayerNorm(model_config['hidden_size'])
        self.attention_heads = model_config['num_heads']
        self.num_layers = model_config['num_layers']
        
    def classify_container(self, 
                         feature_vector: Dict[str, Any]) -> SecurityClassification:
        """
        Implementation of Algorithm 5: Main classification function
        """
        try:
            logger.info("Starting container security classification...")
            
            # Normalize input features
            H0 = self._normalize_features(feature_vector)
            
            # Process through transformer layers
            current_layer = H0
            attention_weights = []
            
            for layer in range(self.num_layers):
                # Multi-head attention
                Q, K, V = self._linear_transform(current_layer)
                attention_output = self._multi_head_attention(Q, K, V)
                attention_weights.append(attention_output['weights'])
                
                # Add & Norm
                attended = self.layer_norm(current_layer + attention_output['values'])
                
                # Feed-forward
                ff_output = self._feed_forward(attended)
                current_layer = self.layer_norm(attended + ff_output)
            
            # Classification head
            security_scores = self._classification_head(current_layer)
            
            # Generate classification result
            classification = self._generate_classification(
                security_scores,
                attention_weights,
                feature_vector
            )
            
            logger.info(f"Classification completed: {classification.level.name}")
            return classification
            
        except Exception as e:
            logger.error(f"Security classification failed: {str(e)}")
            raise

    def _normalize_features(self, features: Dict[str, Any]) -> torch.Tensor:
        """Normalize input features"""
        try:
            normalized = {}
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    normalized[key] = self._min_max_normalize(value)
                elif isinstance(value, dict):
                    normalized[key] = self._normalize_features(value)
                elif isinstance(value, list):
                    normalized[key] = [self._min_max_normalize(v) for v in value]
            return torch.tensor(list(normalized.values()), dtype=torch.float32)
        except Exception as e:
            logger.error(f"Feature normalization failed: {str(e)}")
            raise

    def _linear_transform(self, 
                         input_tensor: torch.Tensor) -> Tuple[torch.Tensor, 
                                                            torch.Tensor, 
                                                            torch.Tensor]:
        """Transform input for attention mechanism"""
        try:
            hidden_size = input_tensor.shape[-1]
            Q = nn.Linear(hidden_size, hidden_size)(input_tensor)
            K = nn.Linear(hidden_size, hidden_size)(input_tensor)
            V = nn.Linear(hidden_size, hidden_size)(input_tensor)
            return Q, K, V
        except Exception as e:
            logger.error(f"Linear transformation failed: {str(e)}")
            raise

    def _multi_head_attention(self, 
                            Q: torch.Tensor, 
                            K: torch.Tensor, 
                            V: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Implement multi-head attention mechanism"""
        try:
            # Scale dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(K.size(-1))
            attention_weights = torch.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention_weights, V)
            
            return {
                'values': attention_output,
                'weights': attention_weights
            }
        except Exception as e:
            logger.error(f"Multi-head attention failed: {str(e)}")
            raise

    def _feed_forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Feed-forward network layer"""
        try:
            ff_size = input_tensor.shape[-1] * 4
            return nn.Sequential(
                nn.Linear(input_tensor.shape[-1], ff_size),
                nn.ReLU(),
                nn.Linear(ff_size, input_tensor.shape[-1])
            )(input_tensor)
        except Exception as e:
            logger.error(f"Feed-forward processing failed: {str(e)}")
            raise

    def _classification_head(self, features: torch.Tensor) -> torch.Tensor:
        """Final classification layer"""
        try:
            return nn.Sequential(
                nn.Linear(features.shape[-1], len(SecurityLevel)),
                nn.Softmax(dim=-1)
            )(features)
        except Exception as e:
            logger.error(f"Classification head processing failed: {str(e)}")
            raise

    def _generate_classification(self,
                               scores: torch.Tensor,
                               attention_weights: List[torch.Tensor],
                               original_features: Dict[str, Any]) -> SecurityClassification:
        """Generate final classification result"""
        try:
            # Get predicted security level
            level_idx = torch.argmax(scores).item()
            confidence = scores[level_idx].item()
            
            # Analyze risk factors using attention weights
            risk_factors = self._analyze_risk_factors(
                attention_weights,
                original_features
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                SecurityLevel(level_idx + 1),
                risk_factors
            )
            
            return SecurityClassification(
                level=SecurityLevel(level_idx + 1),
                confidence=confidence,
                timestamp=datetime.utcnow(),
                features=original_features,
                risk_factors=risk_factors,
                recommendations=recommendations
            )
        except Exception as e:
            logger.error(f"Classification generation failed: {str(e)}")
            raise

class SecurityProfileGenerator:
    """Implementation of Algorithm 6: Security Profile Generation"""

    def __init__(self):
        self.risk_analyzer = RiskAnalyzer()
        self.history_analyzer = HistoryAnalyzer()
        self.context_analyzer = ContextAnalyzer()
        
    def generate_profile(self,
                        classification: SecurityClassification,
                        historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Implementation of Algorithm 6: Main profile generation function
        """
        try:
            logger.info("Starting security profile generation...")
            
            # Compute base risk
            base_risk = self.risk_analyzer.compute_base_risk(classification)
            
            # Analyze historical data
            historical_risk = self.history_analyzer.analyze_history(historical_data)
            
            # Assess context
            context_risk = self.context_analyzer.assess_context()
            
            # Generate risk score
            risk_score = self._compute_risk_score(
                base_risk,
                historical_risk,
                context_risk
            )
            
            # Map risk to security level
            security_level = self._map_risk_to_security(risk_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                security_level,
                risk_score,
                classification
            )
            
            # Create profile
            profile = {
                'risk_score': risk_score,
                'security_level': security_level,
                'base_risk': base_risk,
                'historical_risk': historical_risk,
                'context_risk': context_risk,
                'recommendations': recommendations,
                'timestamp': datetime.utcnow().isoformat(),
                'classification_details': classification.__dict__,
                'profile_id': self._generate_profile_id(classification)
            }
            
            logger.info(f"Generated security profile with ID: {profile['profile_id']}")
            return profile
            
        except Exception as e:
            logger.error(f"Security profile generation failed: {str(e)}")
            raise

    def _compute_risk_score(self,
                           base_risk: float,
                           historical_risk: float,
                           context_risk: float) -> float:
        """Compute weighted risk score"""
        try:
            weights = {
                'base': 0.4,
                'historical': 0.3,
                'context': 0.3
            }
            
            score = (
                weights['base'] * base_risk +
                weights['historical'] * historical_risk +
                weights['historical'] * context_risk
            )
            
            return min(max(score, 0.0), 1.0)  # Normalize to [0,1]
            
        except Exception as e:
            logger.error(f"Risk score computation failed: {str(e)}")
            raise

    def _map_risk_to_security(self, risk_score: float) -> str:
        """Map risk score to security level"""
        if risk_score >= 0.8:
            return "CRITICAL"
        elif risk_score >= 0.6:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"

    def _generate_recommendations(self,
                                security_level: str,
                                risk_score: float,
                                classification: SecurityClassification) -> List[str]:
        """Generate security recommendations"""
        try:
            recommendations = []
            
            # Add base recommendations based on security level
            if security_level == "CRITICAL":
                recommendations.extend([
                    "Implement maximum security measures",
                    "Enable real-time monitoring",
                    "Restrict network access",
                    "Enable audit logging"
                ])
            elif security_level == "HIGH":
                recommendations.extend([
                    "Enable enhanced security measures",
                    "Implement regular monitoring",
                    "Review network policies"
                ])
            
            # Add specific recommendations based on risk factors
            for factor, value in classification.risk_factors.items():
                if value > 0.7:
                    recommendations.append(
                        f"Address high risk factor: {factor}"
                    )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            raise

    def _generate_profile_id(self, 
                           classification: SecurityClassification) -> str:
        """Generate unique profile ID"""
        try:
            data = f"{classification.level.name}:{classification.timestamp}:{classification.confidence}"
            return hashlib.sha256(data.encode()).hexdigest()[:16]
        except Exception as e:
            logger.error(f"Profile ID generation failed: {str(e)}")
            raise

class RiskAnalyzer:
    """Analysis of base risk factors"""
    
    def compute_base_risk(self, 
                         classification: SecurityClassification) -> float:
        """Compute base risk from classification"""
        try:
            # Weight factors
            level_weight = 0.6
            confidence_weight = 0.4
            
            # Normalize security level
            level_risk = classification.level.value / len(SecurityLevel)
            
            # Combine weighted factors
            base_risk = (
                level_weight * level_risk +
                confidence_weight * classification.confidence
            )
            
            return base_risk
            
        except Exception as e:
            logger.error(f"Base risk computation failed: {str(e)}")
            raise

class HistoryAnalyzer:
    """Analysis of historical security data"""
    
    def analyze_history(self, 
                       historical_data: List[Dict[str, Any]]) -> float:
        """Analyze historical security events"""
        try:
            if not historical_data:
                return 0.5  # Default risk for no history
                
            # Calculate trend
            risk_trend = self._calculate_risk_trend(historical_data)
            
            # Analyze patterns
            pattern_risk = self._analyze_patterns(historical_data)
            
            # Combine risks
            historical_risk = 0.7 * risk_trend + 0.3 * pattern_risk
            
            return historical_risk
            
        except Exception as e:
            logger.error(f"Historical analysis failed: {str(e)}")
            raise

class ContextAnalyzer:
    """Analysis of current context and environment"""
    
    def assess_context(self) -> float:
        """Assess current security context"""
        try:
            # Analyze environment
            env_risk = self._analyze_environment()
            
            # Check current threats
            threat_risk = self._assess_current_threats()
            
            # Evaluate network context
            network_risk = self._evaluate_network_context()
            
            # Combine context risks
            context_risk = (
                0.4 * env_risk +
                0.4 * threat_risk +
                0.2 * network_risk
            )
            
            return context_risk
            
        except Exception as e:
            logger.error(f"Context assessment failed: {str(e)}")
            raise