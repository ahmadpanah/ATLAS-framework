import numpy as np
from typing import Dict, List, Tuple
import logging
from ..utils.data_structures import ContainerAttributes, SecurityLevel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerAttributeAnalyzer:
    """
    Implements the Container Attribute Analyzer component from the paper.
    Uses deep learning and feature analysis to classify container security requirements.
    """
    def __init__(self):
        self.security_profiles = {}
        self.feature_weights = {
            'image_size': 0.15,
            'layer_count': 0.10,
            'exposed_ports': 0.20,
            'volume_mounts': 0.15,
            'env_vars': 0.10,
            'resource_limits': 0.15,
            'network_policies': 0.15
        }
        
        # Initialize anomaly detection parameters
        self.isolation_threshold = 0.8
        self.history_window = 100

    def analyze_container(self, container: ContainerAttributes) -> Dict:
        """
        Analyze container attributes and classify security level
        Implements Algorithm 4 from the paper
        """
        try:
            # Extract static and dynamic features
            static_features = self._extract_static_features(container)
            dynamic_features = self._extract_dynamic_features(container)
            
            # Combine features
            features = np.concatenate([static_features, dynamic_features])
            
            # Detect anomalies
            anomaly_score = self._detect_anomalies(features)
            
            # Calculate security scores
            security_scores = self._calculate_security_scores(container)
            
            # Determine final security level
            security_level = self._determine_security_level(
                anomaly_score, 
                security_scores
            )
            
            # Generate security profile
            profile = self._generate_security_profile(
                container,
                security_level,
                security_scores,
                anomaly_score
            )
            
            # Store profile
            self.security_profiles[container.container_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Container analysis failed: {str(e)}")
            raise

    def _extract_static_features(self, container: ContainerAttributes) -> np.ndarray:
        """Extract static features from container attributes"""
        features = [
            container.image_size / 1000,  # Normalize size to MB
            container.layer_count / 10,    # Normalize layer count
            len(container.exposed_ports) / 5,  # Normalize port count
            len(container.volume_mounts) / 3,  # Normalize volume count
            len(container.environment_variables) / 10,  # Normalize env vars
            len(container.network_policies) / 5,  # Normalize network policies
        ]
        return np.array(features)

    def _extract_dynamic_features(self, container: ContainerAttributes) -> np.ndarray:
        """Extract dynamic features from container runtime metrics"""
        # In a real implementation, this would collect runtime metrics
        # Here we're using simplified placeholder values
        features = [
            0.5,  # CPU usage pattern
            0.6,  # Memory usage pattern
            0.4,  # Network activity pattern
            0.5   # I/O operations pattern
        ]
        return np.array(features)

    def _detect_anomalies(self, features: np.ndarray) -> float:
        """
        Detect anomalies using Isolation Forest algorithm
        Implements equation (8) from the paper
        """
        # Simplified anomaly detection using statistical methods
        mean_feature = np.mean(features)
        std_feature = np.std(features)
        z_scores = np.abs((features - mean_feature) / std_feature)
        
        anomaly_score = np.mean(z_scores)
        return min(1.0, anomaly_score)

    def _calculate_security_scores(self, container: ContainerAttributes) -> Dict[str, float]:
        """Calculate security scores for different aspects"""
        scores = {}
        
        # Image security
        scores['image'] = self._calculate_image_security(container)
        
        # Network security
        scores['network'] = self._calculate_network_security(container)
        
        # Resource security
        scores['resource'] = self._calculate_resource_security(container)
        
        # Volume security
        scores['volume'] = self._calculate_volume_security(container)
        
        return scores

    def _calculate_image_security(self, container: ContainerAttributes) -> float:
        """Calculate image security score"""
        image_score = 0.0
        
        # Layer complexity
        layer_score = min(1.0, container.layer_count / 20)
        
        # Image size risk
        size_score = min(1.0, container.image_size / 1000)
        
        image_score = 0.6 * layer_score + 0.4 * size_score
        return image_score

    def _calculate_network_security(self, container: ContainerAttributes) -> float:
        """Calculate network security score"""
        network_score = 0.0
        
        # Exposed ports risk
        port_score = min(1.0, len(container.exposed_ports) / 10)
        
        # Network policies strength
        policy_score = 0.3 if container.network_policies else 1.0
        
        network_score = 0.7 * port_score + 0.3 * policy_score
        return network_score

    def _calculate_resource_security(self, container: ContainerAttributes) -> float:
        """Calculate resource security score"""
        # Check resource limits
        has_cpu_limit = 'cpu' in container.resource_limits
        has_memory_limit = 'memory' in container.resource_limits
        
        if not (has_cpu_limit and has_memory_limit):
            return 1.0  # Highest risk if no resource limits
            
        # Calculate resource risk based on limits
        cpu_risk = min(1.0, container.resource_limits['cpu'] / 4)  # Assume 4 cores is high
        memory_risk = min(1.0, container.resource_limits['memory'] / 8)  # Assume 8GB is high
        
        return 0.5 * (cpu_risk + memory_risk)

    def _calculate_volume_security(self, container: ContainerAttributes) -> float:
        """Calculate volume security score"""
        # Base risk on number of volume mounts
        return min(1.0, len(container.volume_mounts) / 5)

    def _determine_security_level(self, 
                                anomaly_score: float, 
                                security_scores: Dict[str, float]) -> SecurityLevel:
        """Determine final security level based on all scores"""
        # Calculate weighted average of security scores
        weights = {
            'image': 0.3,
            'network': 0.3,
            'resource': 0.2,
            'volume': 0.2
        }
        
        weighted_score = sum(score * weights[aspect] 
                           for aspect, score in security_scores.items())
        
        # Combine with anomaly score
        final_score = 0.7 * weighted_score + 0.3 * anomaly_score
        
        # Map to security level
        if final_score > 0.8:
            return SecurityLevel.CRITICAL
        elif final_score > 0.6:
            return SecurityLevel.HIGH
        elif final_score > 0.4:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.LOW

    def _generate_security_profile(self,
                                 container: ContainerAttributes,
                                 security_level: SecurityLevel,
                                 security_scores: Dict[str, float],
                                 anomaly_score: float) -> Dict:
        """Generate comprehensive security profile"""
        return {
            'container_id': container.container_id,
            'security_level': security_level,
            'security_scores': security_scores,
            'anomaly_score': anomaly_score,
            'recommendations': self._generate_recommendations(security_scores),
            'risk_factors': self._identify_risk_factors(container, security_scores),
            'timestamp': datetime.now()
        }

    def _generate_recommendations(self, security_scores: Dict[str, float]) -> List[str]:
        """Generate security recommendations based on scores"""
        recommendations = []
        
        if security_scores['image'] > 0.6:
            recommendations.append("Consider reducing image size and layer count")
            
        if security_scores['network'] > 0.6:
            recommendations.append("Review and reduce exposed ports")
            recommendations.append("Implement stricter network policies")
            
        if security_scores['resource'] > 0.6:
            recommendations.append("Set appropriate resource limits")
            
        if security_scores['volume'] > 0.6:
            recommendations.append("Review and minimize volume mounts")
            
        return recommendations

    def _identify_risk_factors(self, 
                             container: ContainerAttributes,
                             security_scores: Dict[str, float]) -> List[str]:
        """Identify specific risk factors"""
        risk_factors = []
        
        # Image risks
        if container.layer_count > 10:
            risk_factors.append("High layer count")
        if container.image_size > 1000:
            risk_factors.append("Large image size")
            
        # Network risks
        if len(container.exposed_ports) > 3:
            risk_factors.append("Multiple exposed ports")
        if not container.network_policies:
            risk_factors.append("No network policies defined")
            
        # Resource risks
        if not container.resource_limits:
            risk_factors.append("No resource limits defined")
            
        # Volume risks
        if len(container.volume_mounts) > 2:
            risk_factors.append("Multiple volume mounts")
            
        return risk_factors