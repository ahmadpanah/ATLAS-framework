import logging
from typing import Dict, Optional
import time
from components.migration_controller import MigrationController
from components.federated_learning import FederatedLearningModule
from components.container_analyzer import ContainerAttributeAnalyzer
from components.network_monitor import NetworkConditionMonitor
from components.encryption_engine import AdaptiveEncryptionEngine
from components.optimizer import SecurityPerformanceOptimizer
from utils.data_structures import (
    ContainerAttributes,
    SecurityLevel,
    MigrationRequest,
    NetworkMetrics,
    ContainerMetrics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ATLASFramework:
    def __init__(self):
        """Initialize ATLAS Framework with all components"""
        logger.info("Initializing ATLAS Framework...")
        
        # Initialize all components
        self.controller = MigrationController()
        self.flm = FederatedLearningModule()
        self.caa = ContainerAttributeAnalyzer()
        self.ncm = NetworkConditionMonitor()
        self.aee = AdaptiveEncryptionEngine()
        self.spo = SecurityPerformanceOptimizer()

    def migrate_container(self, 
                         container_id: str,
                         source_cloud: str,
                         destination_cloud: str,
                         security_level: SecurityLevel,
                         priority: int = 1) -> Dict:
        """
        Main method to initiate container migration
        """
        try:
            logger.info(f"Starting migration for container {container_id}")
            
            # Create container attributes
            container_attrs = self._get_container_attributes(container_id)
            
            # Create migration request
            request = MigrationRequest(
                container_id=container_id,
                source_cloud=source_cloud,
                destination_cloud=destination_cloud,
                priority=priority,
                security_level=security_level
            )
            
            # Initiate migration
            result = self.controller.initiate_migration(request)
            migration_id = result['migration_id']
            
            logger.info(f"Migration initiated with ID: {migration_id}")
            return result
            
        except Exception as e:
            logger.error(f"Migration initiation failed: {str(e)}")
            raise

    def monitor_migration(self, migration_id: str, interval: float = 1.0) -> Dict:
        try:
            while True:
                status = self.controller.get_migration_status(migration_id)
                
                if not status:
                    raise ValueError(f"No migration found with ID {migration_id}")
                
                logger.info(
                    f"Migration {migration_id} - State: {status['state']}, "
                    f"Progress: {status['progress']*100:.2f}%"
                )
                
                if status['state'] in ['COMPLETED', 'FAILED']:
                    return status
                    
                time.sleep(interval)
                
        except Exception as e:
            logger.error(f"Migration monitoring failed: {str(e)}")
            raise

    def _get_container_attributes(self, container_id: str) -> ContainerAttributes:
        return ContainerAttributes(
            container_id=container_id,
            image_size=500.0,
            layer_count=5,
            exposed_ports=[80, 443],
            volume_mounts=["/data"],
            environment_variables={"ENV": "prod"},
            resource_limits={
                "cpu": 1.0,
                "memory": 2.0
            },
            network_policies={
                "ingress": "restricted"
            }
        )

def main():
    """Main function demonstrating ATLAS framework usage"""
    try:
        # Initialize ATLAS framework
        atlas = ATLASFramework()
        
        # Example container migration
        container_id = "test_container_123"
        source_cloud = "cloud-a"
        destination_cloud = "cloud-b"
        security_level = SecurityLevel.HIGH
        
        # Start migration
        result = atlas.migrate_container(
            container_id=container_id,
            source_cloud=source_cloud,
            destination_cloud=destination_cloud,
            security_level=security_level
        )
        
        # Monitor migration
        final_status = atlas.monitor_migration(result['migration_id'])
        
        # Print final status
        logger.info(f"Migration completed with status: {final_status}")
        
        # Get migration history
        history = atlas.controller.get_migration_history()
        logger.info(f"Migration history: {history}")
        
        # Cleanup
        atlas.controller.cleanup()
        logger.info("ATLAS framework cleanup completed")
        
    except Exception as e:
        logger.error(f"ATLAS framework error: {str(e)}")
        raise

def example_docker_integration():
    """Example of Docker integration"""
    try:
        import docker
        
        class DockerATLASFramework(ATLASFramework):
            def __init__(self):
                super().__init__()
                self.docker_client = docker.from_env()
                
            def _get_container_attributes(self, container_id: str) -> ContainerAttributes:
                container = self.docker_client.containers.get(container_id)
                config = container.attrs['Config']
                
                return ContainerAttributes(
                    container_id=container_id,
                    image_size=float(container.attrs['Size']) / (1024*1024),
                    layer_count=len(container.attrs['RootFS']['Layers']),
                    exposed_ports=list(config.get('ExposedPorts', {}).keys()),
                    volume_mounts=config.get('Volumes', []),
                    environment_variables=dict(e.split('=') for e in config.get('Env', [])),
                    resource_limits=container.attrs['HostConfig'].get('Resources', {}),
                    network_policies=container.attrs['HostConfig'].get('NetworkMode', {})
                )
        
        # Use Docker-integrated ATLAS
        atlas = DockerATLASFramework()
        
    except ImportError:
        logger.error("Docker SDK not installed. Install with: pip install docker")
        raise

def example_kubernetes_integration():
    """Example of Kubernetes integration"""
    try:
        from kubernetes import client, config
        
        class KubernetesATLASFramework(ATLASFramework):
            def __init__(self):
                super().__init__()
                config.load_kube_config()
                self.k8s_client = client.CoreV1Api()
                
            def _get_container_attributes(self, container_id: str) -> ContainerAttributes:
                # Parse container ID to get pod and container name
                namespace, pod_name, container_name = self._parse_container_id(container_id)
                
                # Get pod details
                pod = self.k8s_client.read_namespaced_pod(
                    name=pod_name,
                    namespace=namespace
                )
                
                # Find container
                container = next(
                    (c for c in pod.spec.containers if c.name == container_name),
                    None
                )
                
                if not container:
                    raise ValueError(f"Container {container_name} not found in pod {pod_name}")
                
                return ContainerAttributes(
                    container_id=container_id,
                    image_size=0.0,
                    layer_count=0,    
                    exposed_ports=[p.container_port for p in container.ports or []],
                    volume_mounts=[v.mount_path for v in container.volume_mounts or []],
                    environment_variables={
                        e.name: e.value for e in container.env or []
                    },
                    resource_limits={
                        "cpu": float(container.resources.limits.get('cpu', 0)),
                        "memory": float(container.resources.limits.get('memory', 0))
                    },
                    network_policies={} 
                )
                
            def _parse_container_id(self, container_id: str) -> tuple:
                """Parse container ID format: namespace/pod-name/container-name"""
                parts = container_id.split('/')
                if len(parts) != 3:
                    raise ValueError(
                        "Invalid container ID format. "
                        "Expected: namespace/pod-name/container-name"
                    )
                return tuple(parts)
        
        # Use Kubernetes-integrated ATLAS
        atlas = KubernetesATLASFramework()
        
    except ImportError:
        logger.error("Kubernetes SDK not installed. Install with: pip install kubernetes")
        raise

if __name__ == "__main__":
    # Run basic example
    main()
    
    # Uncomment to run Docker example
    # example_docker_integration()
    
    # Uncomment to run Kubernetes example
    # example_kubernetes_integration()