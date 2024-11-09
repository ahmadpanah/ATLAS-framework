# components/container_analyzer/feature_extractor.py

import numpy as np
import torch
import docker
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
import networkx as nx
from collections import OrderedDict
import json
import ast
from torch import nn
import torch.nn.functional as F

class AdvancedFeatureExtractor:
    """Advanced feature extraction for container analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.docker_client = docker.from_env()
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        self._initialize_extractors()

    def _initialize_extractors(self):
        """Initialize various feature extractors"""
        try:
            self.static_extractor = StaticFeatureExtractor(self.config)
            self.dynamic_extractor = DynamicFeatureExtractor(self.config)
            self.network_extractor = NetworkFeatureExtractor(self.config)
            self.code_extractor = CodeFeatureExtractor(self.config)
            
        except Exception as e:
            self.logger.error(f"Feature extractor initialization failed: {str(e)}")
            raise

    async def extract_features(self, container_id: str) -> Dict[str, Any]:
        """Extract comprehensive feature set from container"""
        try:
            # Get container info
            container = self.docker_client.containers.get(container_id)
            
            # Extract features
            static_features = await self.static_extractor.extract(container)
            dynamic_features = await self.dynamic_extractor.extract(container)
            network_features = await self.network_extractor.extract(container)
            code_features = await self.code_extractor.extract(container)
            
            # Combine features
            features = {
                'static': static_features,
                'dynamic': dynamic_features,
                'network': network_features,
                'code': code_features,
                'metadata': self._extract_metadata(container)
            }
            
            # Normalize features
            normalized_features = self._normalize_features(features)
            
            return normalized_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize extracted features"""
        try:
            normalized = {}
            for category, feature_set in features.items():
                if isinstance(feature_set, dict):
                    normalized[category] = {
                        k: self.scaler.fit_transform(np.array(v).reshape(-1, 1)).flatten()
                        if isinstance(v, (list, np.ndarray)) else v
                        for k, v in feature_set.items()
                    }
                else:
                    normalized[category] = feature_set
            return normalized
            
        except Exception as e:
            self.logger.error(f"Feature normalization failed: {str(e)}")
            raise

class StaticFeatureExtractor:
    """Extract static features from container"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def extract(self, container: Any) -> Dict[str, Any]:
        """Extract static features"""
        try:
            # Image analysis
            image_features = self._analyze_image(container.image)
            
            # Configuration analysis
            config_features = self._analyze_config(container.attrs['Config'])
            
            # Security profile
            security_features = self._analyze_security_profile(container)
            
            # Dependency analysis
            dependency_features = self._analyze_dependencies(container)
            
            return {
                'image': image_features,
                'config': config_features,
                'security': security_features,
                'dependencies': dependency_features
            }
            
        except Exception as e:
            self.logger.error(f"Static feature extraction failed: {str(e)}")
            raise

    def _analyze_image(self, image: Any) -> Dict[str, Any]:
        """Analyze container image"""
        return {
            'size': image.attrs['Size'],
            'layers': len(image.history()),
            'age': (datetime.now() - 
                   datetime.strptime(image.attrs['Created'], 
                   '%Y-%m-%dT%H:%M:%S.%fZ')).days,
            'os': image.attrs['Os'],
            'architecture': image.attrs['Architecture']
        }

    def _analyze_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze container configuration"""
        return {
            'env_vars': len(config.get('Env', [])),
            'exposed_ports': len(config.get('ExposedPorts', {})),
            'volumes': len(config.get('Volumes', {})),
            'labels': len(config.get('Labels', {})),
            'entrypoint': bool(config.get('Entrypoint')),
            'user': config.get('User', 'root')
        }

    def _analyze_security_profile(self, container: Any) -> Dict[str, Any]:
        """Analyze security profile"""
        return {
            'privileged': container.attrs['HostConfig']['Privileged'],
            'capabilities': len(container.attrs['HostConfig']['CapAdd'] or []),
            'readonly_rootfs': container.attrs['HostConfig']['ReadonlyRootfs'],
            'security_opt': len(container.attrs['HostConfig']['SecurityOpt'] or [])
        }

    def _analyze_dependencies(self, container: Any) -> Dict[str, Any]:
        """Analyze container dependencies"""
        # Implementation depends on container type and available metadata
        return {}

class DynamicFeatureExtractor:
    """Extract dynamic features from container"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window_size = config.get('window_size', 100)
        self.logger = logging.getLogger(__name__)

    async def extract(self, container: Any) -> Dict[str, Any]:
        """Extract dynamic features"""
        try:
            # Resource usage
            resource_metrics = self._collect_resource_metrics(container)
            
            # Network activity
            network_metrics = self._collect_network_metrics(container)
            
            # Process activity
            process_metrics = self._collect_process_metrics(container)
            
            # I/O activity
            io_metrics = self._collect_io_metrics(container)
            
            # Time series analysis
            time_series = self._analyze_time_series(
                resource_metrics,
                network_metrics,
                process_metrics,
                io_metrics
            )
            
            return {
                'resource': resource_metrics,
                'network': network_metrics,
                'process': process_metrics,
                'io': io_metrics,
                'time_series': time_series
            }
            
        except Exception as e:
            self.logger.error(f"Dynamic feature extraction failed: {str(e)}")
            raise

    def _collect_resource_metrics(self, container: Any) -> Dict[str, float]:
        """Collect resource usage metrics"""
        stats = container.stats(stream=False)
        return {
            'cpu_usage': self._calculate_cpu_percent(stats),
            'memory_usage': self._calculate_memory_percent(stats),
            'memory_limit': stats['memory_stats']['limit'],
            'rx_bytes': stats['networks']['eth0']['rx_bytes'],
            'tx_bytes': stats['networks']['eth0']['tx_bytes']
        }

    def _calculate_cpu_percent(self, stats: Dict[str, Any]) -> float:
        """Calculate CPU percentage"""
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                   stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                      stats['precpu_stats']['system_cpu_usage']
        return (cpu_delta / system_delta) * 100.0

    def _calculate_memory_percent(self, stats: Dict[str, Any]) -> float:
        """Calculate memory percentage"""
        return (stats['memory_stats']['usage'] / 
                stats['memory_stats']['limit']) * 100.0

class NetworkFeatureExtractor:
    """Extract network features from container"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def extract(self, container: Any) -> Dict[str, Any]:
        """Extract network features"""
        try:
            # Network connections
            connections = self._analyze_connections(container)
            
            # Traffic patterns
            traffic = self._analyze_traffic(container)
            
            # Protocol analysis
            protocols = self._analyze_protocols(container)
            
            # Network graph
            graph = self._create_network_graph(connections)
            
            return {
                'connections': connections,
                'traffic': traffic,
                'protocols': protocols,
                'graph_metrics': self._calculate_graph_metrics(graph)
            }
            
        except Exception as e:
            self.logger.error(f"Network feature extraction failed: {str(e)}")
            raise

class CodeFeatureExtractor:
    """Extract code-level features from container"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    async def extract(self, container: Any) -> Dict[str, Any]:
        """Extract code features"""
        try:
            # Static code analysis
            code_metrics = self._analyze_code(container)
            
            # Dependency analysis
            dependencies = self._analyze_dependencies(container)
            
            # Security patterns
            security_patterns = self._analyze_security_patterns(container)
            
            return {
                'code_metrics': code_metrics,
                'dependencies': dependencies,
                'security_patterns': security_patterns
            }
            
        except Exception as e:
            self.logger.error(f"Code feature extraction failed: {str(e)}")
            raise