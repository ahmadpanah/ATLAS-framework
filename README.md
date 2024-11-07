# ATLAS Framework

  

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)

[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

[![Kubernetes](https://img.shields.io/badge/kubernetes-%23326ce5.svg?style=flat&logo=kubernetes&logoColor=white)](https://kubernetes.io/)

  

ATLAS (Adaptive Transfer and Learning-based Secure Container Migration Framework) is a comprehensive framework for secure container migration across cloud environments. It provides intelligent security adaptation, performance optimization, and threat monitoring during container migration.

  

## Features

  

### Core Components

  

-  **Federated Learning Module (FLM)**

	- Collaborative threat intelligence sharing

	- Privacy-preserving learning

	- Distributed model training

	- Policy generation

  

-  **Container Attribute Analyzer (CAA)**

	- Feature extraction

	- Deep learning classification

	- Security profiling

	- Real-time analysis

  

-  **Network Condition Monitor (NCM)**

	- Real-time metrics collection

	- Performance analysis

	- Status management

	- Network quality assessment

  

-  **Adaptive Encryption Engine (AEE)**

	- Dynamic algorithm selection

	- Parameter optimization

	- State-based encryption management

	- Performance monitoring

  

-  **Security-Performance Optimizer (SPO)**

	- Trade-off analysis

	- Resource management

	- Performance monitoring

	- Adaptive optimization

  

### Key Features

  

- Federated Learning-based Security

- Adaptive Security Measures

- Performance Optimization

- Real-time Monitoring

- Secure Migration

- Resource Optimization

  

## System Requirements

  

### Software Requirements

- Python 3.12+

- Docker 20.10+

- Kubernetes 1.21+

- MongoDB 4.4+

  

### Hardware Recommendations

- CPU: 4+ cores

- RAM: 8GB+

- Storage: 100GB+

- Network: 1Gbps+

  

## Installation


 1. Clone the repository:

	    git clone https://github.com/ahmadpanah/ATLAS-framework.git
        
	    cd ATLAS-framework


 2. Create and activate virtual environment:

## Linux/Mac

    python -m venv venv
    
    source venv/bin/activate


## Windows

    python -m venv venv
    
    .\venv\Scripts\activate

 3. install required packages:

	    pip install -r requirements.txt

 4. Configure environment variables:

## Linux/Mac

export ATLAS_ENV=development

export ATLAS_LOG_LEVEL=INFO

  

## Windows

set ATLAS_ENV=development

set ATLAS_LOG_LEVEL=INFO

## Configuration

The framework uses a hierarchical configuration system. Core configurations are located in the `config/` directory:


```
# Example configuration override
from config import default_config

custom_config = {
    'security': {
        'levels': {
            'HIGH': {
                'encryption': 'AES-256-GCM',
                'key_rotation': 21600,
                'monitoring': 'continuous'
            }
        }
    }
}

# Merge configurations
config = {**default_config, **custom_config}

```

## Usage

### Basic Usage

```
from atlas import ATLASFramework
from utils.data_structures import MigrationRequest, SecurityLevel

# Initialize framework
atlas = ATLASFramework(config)

# Create migration request
request = MigrationRequest(
    container_id="container123",
    source_cloud="cloud-a",
    destination_cloud="cloud-b",
    priority=1,
    security_level=SecurityLevel.HIGH
)

# Perform migration
result = await atlas.migrate_container(request)

```

### Advanced Usage

```
# Custom security profile
from components.container_analyzer import SecurityProfiler

profiler = SecurityProfiler(config['security'])
profile = profiler.generate_profile(
    container_id="container123",
    classification_results=classification,
    features=features
)

# Network monitoring
from components.network_monitor import MetricsCollector

collector = MetricsCollector(config['monitoring'])
collector.start_collection()
metrics = collector.get_current_metrics()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

-   Seyed Hossein Ahmadpanah - [h.ahmadpanah@iau.ac.ir](mailto:h.ahmadpanah@iau.ac.ir)
-   Project Link: [https://github.com/ahmadpanah/ATLAS-framework](https://github.com/ahmadpanah/ATLAS-framework)