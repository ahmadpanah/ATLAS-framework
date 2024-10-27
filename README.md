# ATLAS: Adaptive Transfer and Learning-based Secure Container Migration Framework

ATLAS is a Python framework for secure container migration across cloud environments. It provides intelligent security adaptation, performance optimization, and threat monitoring during container migration.

## Features

- **Federated Learning-based Security**: Collaborative threat intelligence sharing across cloud providers
- **Adaptive Security**: Dynamic security measures based on container sensitivity and network conditions
- **Performance Optimization**: Balance between security requirements and performance constraints
- **Real-time Monitoring**: Continuous network and container monitoring
- **Secure Migration**: End-to-end encrypted container transfer
- **Resource Optimization**: Intelligent resource allocation based on security needs

## Requirements

- Python 3.8+
- NumPy
- Cryptography
- SciPy

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ahmadpanah/atlas-framework.git
   cd atlas-framework

2.  Create a virtual environment and activate it:

    
    ```
    # Linux/Mac
    python -m venv venv
    source venv/bin/activate
    ```
    
3.  Install required packages:
    
    ```
    pip install -r requirements.txt
    ```

## Basic Usage Example

Here's a basic example of how to use the ATLAS framework:
```
from atlas.components.migration_controller import MigrationController
from atlas.utils.data_structures import (
    MigrationRequest,
    SecurityLevel,
    ContainerAttributes
)

# Initialize the controller
controller = MigrationController()

# Create container attributes
container_attrs = ContainerAttributes(
    container_id="container123",
    image_size=500.0,  # MB
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

# Create migration request
request = MigrationRequest(
    container_id="container123",
    source_cloud="cloud-a",
    destination_cloud="cloud-b",
    priority=1,
    security_level=SecurityLevel.HIGH
)

# Initiate migration
result = controller.initiate_migration(request)
migration_id = result['migration_id']

# Monitor migration status
import time

while True:
    status = controller.get_migration_status(migration_id)
    print(f"Migration status: {status['state']}")
    print(f"Progress: {status['progress']*100:.2f}%")
    
    if status['state'] in ['COMPLETED', 'FAILED']:
        break
    
    time.sleep(1)

# Get migration history
history = controller.get_migration_history()
print("Migration History:", history)

# Cleanup
controller.cleanup()
```

## Advanced Usage Examples

### Docker Integration

```
import docker
from atlas.components.migration_controller import MigrationController

class DockerMigrationController(MigrationController):
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
```

### Monitoring Example

```
# Get network quality
network_quality = controller.ncm.analyze_network_quality()
print("Network Quality:", network_quality)

# Get optimization history
optimization_history = controller.spo.get_optimization_history("container123")
print("Optimization History:", optimization_history)

# Get encryption statistics
encryption_stats = controller.aee.get_encryption_stats("container123")
print("Encryption Stats:", encryption_stats)
```

### Custom Security Configuration

```
from atlas.components.encryption_engine import AdaptiveEncryptionEngine

aee = AdaptiveEncryptionEngine()
# Configure encryption parameters for different security levels
aee.encryption_configs[SecurityLevel.HIGH] = {
    'algorithm': 'AES',
    'key_size': 256,
    'mode': 'GCM',
    'iterations': 200000,
    'memory_hard': True
}
```

## Debugging

Enable detailed logging:

```
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## License

This project is licensed under the MIT License.
