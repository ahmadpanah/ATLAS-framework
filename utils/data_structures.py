
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

class SecurityLevel(Enum):
    """Security levels for container classification"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EncryptionMode(Enum):
    """Supported encryption modes"""
    GCM = "GCM"
    CBC = "CBC"
    CTR = "CTR"
    OCB = "OCB"

@dataclass
class ContainerAttributes:
    """Container attributes data structure"""
    container_id: str
    image_size: float
    layer_count: int
    exposed_ports: List[int]
    volume_mounts: List[str]
    environment_variables: Dict[str, str]
    resource_limits: Dict[str, float]
    network_policies: Dict[str, str]
    security_level: SecurityLevel
    created_at: datetime
    updated_at: datetime

@dataclass
class MigrationRequest:
    """Container migration request"""
    container_id: str
    source_cloud: str
    destination_cloud: str
    priority: int
    security_level: SecurityLevel
    attributes: ContainerAttributes
    requirements: Dict[str, Any]
    timestamp: datetime

@dataclass
class NetworkCondition:
    """Network condition data"""
    latency: float
    bandwidth: float
    packet_loss: float
    jitter: float
    stability: float
    timestamp: datetime

@dataclass
class SecurityPolicy:
    """Security policy definition"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]
    priority: int
    security_level: SecurityLevel
    created_at: datetime
    updated_at: datetime
    enabled: bool

@dataclass
class EncryptionConfig:
    """Encryption configuration"""
    algorithm: str
    mode: EncryptionMode
    key_size: int
    iv_size: Optional[int]
    tag_size: Optional[int]
    iterations: int
    memory_hard: bool
    parameters: Dict[str, Any]

@dataclass
class MigrationStatus:
    """Migration status information"""
    migration_id: str
    container_id: str
    state: str
    progress: float
    current_phase: str
    security_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    errors: List[Dict[str, Any]]
    start_time: datetime
    update_time: datetime

@dataclass
class ResourceAllocation:
    """Resource allocation information"""
    container_id: str
    cpu_allocation: float
    memory_allocation: float
    disk_allocation: float
    network_allocation: float
    priority: int
    timestamp: datetime

@dataclass
class PerformanceMetrics:
    """Performance metrics data"""
    cpu_usage: float
    memory_usage: float
    disk_io: Dict[str, float]
    network_io: Dict[str, float]
    latency: float
    throughput: float
    timestamp: datetime

@dataclass
class SecurityMetrics:
    """Security metrics data"""
    encryption_strength: float
    vulnerability_score: float
    threat_level: float
    compliance_score: float
    integrity_score: float
    timestamp: datetime

@dataclass
class FederatedLearningMetrics:
    """Federated learning metrics"""
    model_version: str
    accuracy: float
    loss: float
    training_round: int
    participants: int
    convergence_rate: float
    timestamp: datetime

class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

class DataStructureValidator:
    """Validator for data structures"""
    
    @staticmethod
    def validate_container_attributes(attrs: ContainerAttributes) -> bool:
        """Validate container attributes"""
        try:
            if not attrs.container_id or len(attrs.container_id) < 12:
                raise DataValidationError("Invalid container ID")
                
            if attrs.image_size <= 0:
                raise DataValidationError("Invalid image size")
                
            if attrs.layer_count <= 0:
                raise DataValidationError("Invalid layer count")
                
            if not all(isinstance(port, int) and port > 0 
                      for port in attrs.exposed_ports):
                raise DataValidationError("Invalid exposed ports")
                
            return True
            
        except Exception as e:
            raise DataValidationError(f"Validation failed: {str(e)}")

    @staticmethod
    def validate_migration_request(request: MigrationRequest) -> bool:
        """Validate migration request"""
        try:
            if not request.container_id or not request.source_cloud or \
               not request.destination_cloud:
                raise DataValidationError("Missing required fields")
                
            if request.priority < 0:
                raise DataValidationError("Invalid priority")
                
            DataStructureValidator.validate_container_attributes(
                request.attributes
            )
            
            return True
            
        except Exception as e:
            raise DataValidationError(f"Validation failed: {str(e)}")

    @staticmethod
    def validate_security_policy(policy: SecurityPolicy) -> bool:
        """Validate security policy"""
        try:
            if not policy.policy_id or not policy.name:
                raise DataValidationError("Missing required fields")
                
            if not policy.rules:
                raise DataValidationError("Empty policy rules")
                
            if policy.priority < 0:
                raise DataValidationError("Invalid priority")
                
            return True
            
        except Exception as e:
            raise DataValidationError(f"Validation failed: {str(e)}")

    @staticmethod
    def validate_encryption_config(config: EncryptionConfig) -> bool:
        """Validate encryption configuration"""
        try:
            if not config.algorithm:
                raise DataValidationError("Missing algorithm")
                
            if config.key_size not in [128, 192, 256]:
                raise DataValidationError("Invalid key size")
                
            if config.mode == EncryptionMode.GCM and not config.tag_size:
                raise DataValidationError("Missing tag size for GCM mode")
                
            return True
            
        except Exception as e:
            raise DataValidationError(f"Validation failed: {str(e)}")