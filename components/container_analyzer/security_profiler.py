import torch
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
import logging
from datetime import datetime

@dataclass
class SecurityProfile:
    """Security profile data structure"""
    container_id: str
    risk_score: float
    security_level: str
    vulnerabilities: List[Dict[str, Any]]
    recommendations: List[str]
    timestamp: datetime
    confidence: float

class SecurityProfiler:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Security Profiler with configuration
        
        Args:
            config: Configuration dictionary containing:
                - risk_thresholds: Risk level thresholds
                - security_levels: Security level definitions
                - update_frequency: Profile update frequency
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize profiler components"""
        self.risk_thresholds = self.config['risk_thresholds']
        self.security_levels = self.config['security_levels']
        self.profile_history = {}
        
        # Initialize fuzzy logic system
        self._initialize_fuzzy_system()

    def generate_profile(self, 
                        container_id: str,
                        classification_results: Dict[str, Any],
                        features: Dict[str, Any]) -> SecurityProfile:
        """
        Generate security profile for container
        
        Args:
            container_id: Container identifier
            classification_results: Results from classifier
            features: Container features
            
        Returns:
            SecurityProfile object
        """
        try:
            # Compute risk score
            risk_score = self._compute_risk_score(
                classification_results,
                features
            )
            
            # Determine security level
            security_level = self._determine_security_level(risk_score)
            
            # Identify vulnerabilities
            vulnerabilities = self._identify_vulnerabilities(
                classification_results,
                features
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                vulnerabilities,
                security_level
            )
            
            # Create profile
            profile = SecurityProfile(
                container_id=container_id,
                risk_score=risk_score,
                security_level=security_level,
                vulnerabilities=vulnerabilities,
                recommendations=recommendations,
                timestamp=datetime.now(),
                confidence=classification_results['confidence'].mean()
            )
            
            # Update history
            self._update_profile_history(container_id, profile)
            
            return profile
            
        except Exception as e:
            self.logger.error(f"Profile generation failed: {str(e)}")
            raise

    def _compute_risk_score(self,
                           classification_results: Dict[str, Any],
                           features: Dict[str, Any]) -> float:
        """Compute risk score using fuzzy logic"""
        try:
            # Extract relevant metrics
            threat_level = self._evaluate_threat_level(classification_results)
            vulnerability_score = self._evaluate_vulnerabilities(features)
            exposure_level = self._evaluate_exposure(features)
            
            # Apply fuzzy rules
            risk_score = self.fuzzy_system.evaluate(
                threat_level,
                vulnerability_score,
                exposure_level
            )
            
            return float(risk_score)
            
        except Exception as e:
            self.logger.error(f"Risk score computation failed: {str(e)}")
            raise

    def _determine_security_level(self, risk_score: float) -> str:
        """Determine security level based on risk score"""
        for level, threshold in self.risk_thresholds.items():
            if risk_score <= threshold:
                return level
        return "CRITICAL"

    def _identify_vulnerabilities(self,
                                classification_results: Dict[str, Any],
                                features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify security vulnerabilities"""
        vulnerabilities = []
        
        # Check configuration vulnerabilities
        vulns = self._check_config_vulnerabilities(features)
        vulnerabilities.extend(vulns)
        
        # Check runtime vulnerabilities
        vulns = self._check_runtime_vulnerabilities(features)
        vulnerabilities.extend(vulns)
        
        # Check network vulnerabilities
        vulns = self._check_network_vulnerabilities(features)
        vulnerabilities.extend(vulns)
        
        return vulnerabilities

    def _generate_recommendations(self,
                                vulnerabilities: List[Dict[str, Any]],
                                security_level: str) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        # Generate general recommendations
        recommendations.extend(
            self._generate_general_recommendations(security_level)
        )
        
        # Generate vulnerability-specific recommendations
        recommendations.extend(
            self._generate_vulnerability_recommendations(vulnerabilities)
        )
        
        return recommendations

    def _initialize_fuzzy_system(self):
        """Initialize fuzzy logic system"""
        self.fuzzy_system = FuzzyLogicSystem(
            input_vars=['threat', 'vulnerability', 'exposure'],
            output_vars=['risk'],
            rules=self.config.get('fuzzy_rules', [])
        )

    def _update_profile_history(self, 
                              container_id: str,
                              profile: SecurityProfile):
        """Update profile history"""
        if container_id not in self.profile_history:
            self.profile_history[container_id] = []
        
        self.profile_history[container_id].append(profile)
        
        # Maintain history size
        max_history = self.config.get('max_history_size', 100)
        if len(self.profile_history[container_id]) > max_history:
            self.profile_history[container_id].pop(0)

    def get_profile_history(self, 
                          container_id: str,
                          limit: int = None) -> List[SecurityProfile]:
        """Get profile history for container"""
        history = self.profile_history.get(container_id, [])
        if limit:
            return history[-limit:]
        return history