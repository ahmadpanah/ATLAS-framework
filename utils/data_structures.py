from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from datetime import datetime

class SecurityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ContainerMetrics:
    container_id: str
    cpu_usage: float
    memory_usage: float
    network_throughput: float
    disk_io: float

@dataclass
class NetworkMetrics:
    bandwidth: float  # Mbps
    latency: float   # ms
    packet_loss: float  # percentage
    jitter: float    # ms
    timestamp: datetime = datetime.now()

@dataclass
class ContainerAttributes:
    container_id: str
    image_size: float
    layer_count: int
    exposed_ports: List[int]
    volume_mounts: List[str]
    environment_variables: Dict[str, str]
    resource_limits: Dict[str, float]
    network_policies: Dict[str, str]

@dataclass
class MigrationRequest:
    container_id: str
    source_cloud: str
    destination_cloud: str
    priority: int
    security_level: SecurityLevel