
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class SecurityPolicy:
    """Data class for security policies"""
    policy_id: str
    policy_type: str  # base, contextual, or dynamic
    rules: List[Dict[str, Any]]
    priority: int
    created_at: datetime
    updated_at: datetime
    effectiveness_score: float = 0.0
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0

class PolicyGenerator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Policy Generator with configuration parameters
        
        Args:
            config: Configuration dictionary containing:
                - learning_rate: Learning rate for policy optimization
                - risk_threshold: Threshold for risk assessment
                - update_frequency: Frequency of policy updates
                - base_policies: List of base security policies
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize policy generation components"""
        self.base_policies = self.config.get('base_policies', [])
        self.contextual_policies = []
        self.dynamic_policies = []
        self.policy_history = []
        self.effectiveness_metrics = {}
        
        # Initialize policy weights
        self.policy_weights = torch.ones(len(self.base_policies))
        self.policy_weights /= self.policy_weights.sum()

    def generate_policies(self, 
                        threat_data: Dict[str, Any],
                        context: Dict[str, Any]) -> List[SecurityPolicy]:
        """
        Generate security policies based on threat data and context
        
        Args:
            threat_data: Dictionary containing threat intelligence
            context: Current operational context
            
        Returns:
            List of generated security policies
        """
        try:
            # Generate policies at each level
            base = self._generate_base_policies(threat_data)
            contextual = self._generate_contextual_policies(threat_data, context)
            dynamic = self._generate_dynamic_policies(threat_data, context)
            
            # Combine and prioritize policies
            combined_policies = self._combine_policies(base, contextual, dynamic)
            
            # Validate and optimize policies
            optimized_policies = self._optimize_policies(combined_policies)
            
            return optimized_policies
            
        except Exception as e:
            self.logger.error(f"Policy generation failed: {str(e)}")
            return self.base_policies  # Fallback to base policies

    def _generate_base_policies(self, threat_data: Dict[str, Any]) -> List[SecurityPolicy]:
        """Generate base security policies"""
        base_policies = []
        try:
            # Extract fundamental security requirements
            security_requirements = self._extract_security_requirements(threat_data)
            
            for req in security_requirements:
                policy = SecurityPolicy(
                    policy_id=f"base_{datetime.now().timestamp()}",
                    policy_type="base",
                    rules=self._create_base_rules(req),
                    priority=100,  # Highest priority for base policies
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                base_policies.append(policy)
                
        except Exception as e:
            self.logger.error(f"Base policy generation failed: {str(e)}")
            
        return base_policies

    def _generate_contextual_policies(self, 
                                    threat_data: Dict[str, Any],
                                    context: Dict[str, Any]) -> List[SecurityPolicy]:
        """Generate context-aware security policies"""
        contextual_policies = []
        try:
            # Analyze context for policy generation
            context_features = self._extract_context_features(context)
            
            # Generate policies based on context
            for feature in context_features:
                policy = SecurityPolicy(
                    policy_id=f"ctx_{datetime.now().timestamp()}",
                    policy_type="contextual",
                    rules=self._create_contextual_rules(feature, threat_data),
                    priority=50,  # Medium priority
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                contextual_policies.append(policy)
                
        except Exception as e:
            self.logger.error(f"Contextual policy generation failed: {str(e)}")
            
        return contextual_policies

    def _generate_dynamic_policies(self,
                                 threat_data: Dict[str, Any],
                                 context: Dict[str, Any]) -> List[SecurityPolicy]:
        """Generate dynamic security policies"""
        dynamic_policies = []
        try:
            # Analyze current threats and trends
            threat_patterns = self._analyze_threat_patterns(threat_data)
            
            # Generate dynamic policies based on current threats
            for pattern in threat_patterns:
                policy = SecurityPolicy(
                    policy_id=f"dyn_{datetime.now().timestamp()}",
                    policy_type="dynamic",
                    rules=self._create_dynamic_rules(pattern, context),
                    priority=25,  # Lower priority but more specific
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
                dynamic_policies.append(policy)
                
        except Exception as e:
            self.logger.error(f"Dynamic policy generation failed: {str(e)}")
            
        return dynamic_policies

    def _optimize_policies(self, 
                         policies: List[SecurityPolicy]) -> List[SecurityPolicy]:
        """Optimize policies using risk-weighted utility function"""
        try:
            # Calculate utility scores
            utilities = []
            for policy in policies:
                utility = self._calculate_utility(policy)
                utilities.append(utility)
            
            # Normalize utilities
            utilities = torch.tensor(utilities)
            utilities = utilities / utilities.sum()
            
            # Update policy weights
            self.policy_weights = self._update_weights(utilities)
            
            # Filter and rank policies
            optimized_policies = self._rank_policies(policies, utilities)
            
            return optimized_policies
            
        except Exception as e:
            self.logger.error(f"Policy optimization failed: {str(e)}")
            return policies

    def _calculate_utility(self, policy: SecurityPolicy) -> float:
        """Calculate utility score for a policy"""
        try:
            # Components of utility function
            security_score = self._evaluate_security_effectiveness(policy)
            overhead_score = self._evaluate_overhead(policy)
            risk_score = self._evaluate_risk_mitigation(policy)
            
            # Weighted combination
            utility = (
                self.config.get('security_weight', 0.5) * security_score +
                self.config.get('overhead_weight', 0.3) * (1 - overhead_score) +
                self.config.get('risk_weight', 0.2) * risk_score
            )
            
            return float(utility)
            
        except Exception as e:
            self.logger.error(f"Utility calculation failed: {str(e)}")
            return 0.0

    def update_policy_effectiveness(self, 
                                  policy_id: str,
                                  metrics: Dict[str, float]):
        """Update policy effectiveness metrics"""
        try:
            if policy_id in self.effectiveness_metrics:
                policy_metrics = self.effectiveness_metrics[policy_id]
                
                # Update metrics with exponential moving average
                alpha = self.config.get('metrics_smoothing', 0.1)
                for metric, value in metrics.items():
                    if metric in policy_metrics:
                        policy_metrics[metric] = (
                            alpha * value + 
                            (1 - alpha) * policy_metrics[metric]
                        )
                    else:
                        policy_metrics[metric] = value
                        
            else:
                self.effectiveness_metrics[policy_id] = metrics
                
        except Exception as e:
            self.logger.error(f"Policy effectiveness update failed: {str(e)}")

    def _evaluate_security_effectiveness(self, policy: SecurityPolicy) -> float:
        """Evaluate security effectiveness of a policy"""
        if policy.policy_id in self.effectiveness_metrics:
            metrics = self.effectiveness_metrics[policy.policy_id]
            
            # Calculate effectiveness score
            effectiveness = (
                (1 - metrics.get('false_positive_rate', 0.0)) * 0.4 +
                (1 - metrics.get('false_negative_rate', 0.0)) * 0.6
            )
            
            return float(effectiveness)
        return 0.5  # Default score for new policies

    def _evaluate_overhead(self, policy: SecurityPolicy) -> float:
        """Evaluate computational overhead of a policy"""
        try:
            # Calculate resource requirements
            cpu_overhead = self._estimate_cpu_overhead(policy.rules)
            memory_overhead = self._estimate_memory_overhead(policy.rules)
            network_overhead = self._estimate_network_overhead(policy.rules)
            
            # Weighted combination
            total_overhead = (
                0.4 * cpu_overhead +
                0.3 * memory_overhead +
                0.3 * network_overhead
            )
            
            return float(total_overhead)
            
        except Exception as e:
            self.logger.error(f"Overhead evaluation failed: {str(e)}")
            return 1.0  # Conservative estimate

    def _evaluate_risk_mitigation(self, policy: SecurityPolicy) -> float:
        """Evaluate risk mitigation effectiveness of a policy"""
        try:
            # Calculate risk mitigation score
            coverage = self._calculate_threat_coverage(policy.rules)
            specificity = self._calculate_rule_specificity(policy.rules)
            adaptability = self._calculate_adaptability(policy)
            
            # Weighted combination
            risk_score = (
                0.4 * coverage +
                0.3 * specificity +
                0.3 * adaptability
            )
            
            return float(risk_score)
            
        except Exception as e:
            self.logger.error(f"Risk evaluation failed: {str(e)}")
            return 0.0

    def _update_weights(self, utilities: torch.Tensor) -> torch.Tensor:
        """Update policy weights using gradient descent"""
        try:
            # Calculate gradient
            gradient = self._calculate_weight_gradient(utilities)
            
            # Update weights with learning rate
            lr = self.config.get('learning_rate', 0.01)
            new_weights = self.policy_weights + lr * gradient
            
            # Normalize weights
            new_weights = torch.clamp(new_weights, min=0.0)
            new_weights = new_weights / new_weights.sum()
            
            return new_weights
            
        except Exception as e:
            self.logger.error(f"Weight update failed: {str(e)}")
            return self.policy_weights

    def get_policy_stats(self) -> Dict[str, Any]:
        """Get statistics about current policies"""
        return {
            'total_policies': len(self.base_policies) + 
                            len(self.contextual_policies) + 
                            len(self.dynamic_policies),
            'effectiveness_metrics': self.effectiveness_metrics,
            'policy_weights': self.policy_weights.tolist(),
            'update_history': self.policy_history
        }
