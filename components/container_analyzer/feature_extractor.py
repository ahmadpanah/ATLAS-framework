python

Copy
# components/container_analyzer/feature_extractor.py

import torch
import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import statsmodels.api as sm
from scipy.fft import fft
import docker

@dataclass
class ContainerFeatures:
    """Container feature data structure"""
    static_features: Dict[str, Any]
    dynamic_features: Dict[str, Any]
    timestamp: datetime
    container_id: str

class FeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Feature Extractor with configuration
        
        Args:
            config: Configuration dictionary containing:
                - sampling_rate: Rate for dynamic feature collection
                - window_size: Size of monitoring window
                - feature_dimensions: Feature vector dimensions
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.docker_client = docker.from_env()
        self._initialize_components()

    def _initialize_components(self):
        """Initialize feature extraction components"""
        self.window_size = self.config.get('window_size', 100)
        self.sampling_rate = self.config.get('sampling_rate', 1.0)
        self.feature_buffer = []
        
    def extract_features(self, container_id: str) -> ContainerFeatures:
        """
        Extract both static and dynamic features from container
        
        Args:
            container_id: Docker container ID
            
        Returns:
            ContainerFeatures object containing extracted features
        """
        try:
            # Extract static features
            static_features = self._extract_static_features(container_id)
            
            # Collect dynamic features
            dynamic_features = self._extract_dynamic_features(container_id)
            
            # Combine features
            features = ContainerFeatures(
                static_features=static_features,
                dynamic_features=dynamic_features,
                timestamp=datetime.now(),
                container_id=container_id
            )
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {str(e)}")
            raise

    def _extract_static_features(self, container_id: str) -> Dict[str, Any]:
        """Extract static container features"""
        try:
            container = self.docker_client.containers.get(container_id)
            config = container.attrs['Config']
            
            static_features = {
                'image_features': self._analyze_image(container.image),
                'config_features': self._parse_configuration(config),
                'dependency_features': self._analyze_dependencies(container),
                'permission_features': self._analyze_permissions(container),
                'network_features': self._analyze_network_config(container)
            }
            
            return static_features
            
        except Exception as e:
            self.logger.error(f"Static feature extraction failed: {str(e)}")
            raise

    def _extract_dynamic_features(self, container_id: str) -> Dict[str, Any]:
        """Extract dynamic container features"""
        try:
            # Initialize metrics buffer
            metrics_buffer = []
            
            # Collect metrics over window
            for _ in range(self.window_size):
                metrics = self._collect_metrics(container_id)
                processed_metrics = self._process_metrics(metrics)
                metrics_buffer.append(processed_metrics)
                
            # Process time series features
            dynamic_features = {
                'time_series': self._process_time_series(metrics_buffer),
                'statistical': self._compute_statistical_features(metrics_buffer),
                'frequency': self._compute_frequency_features(metrics_buffer)
            }
            
            return dynamic_features
            
        except Exception as e:
            self.logger.error(f"Dynamic feature extraction failed: {str(e)}")
            raise

    def _analyze_image(self, image) -> Dict[str, Any]:
        """Analyze container image features"""
        return {
            'size': image.attrs['Size'],
            'layers': len(image.attrs['RootFS']['Layers']),
            'created': image.attrs['Created'],
            'architecture': image.attrs['Architecture']
        }

    def _parse_configuration(self, config: Dict) -> Dict[str, Any]:
        """Parse container configuration"""
        return {
            'environment': self._process_env_vars(config.get('Env', [])),
            'exposed_ports': list(config.get('ExposedPorts', {}).keys()),
            'volumes': list(config.get('Volumes', {}).keys()),
            'labels': config.get('Labels', {}),
            'entrypoint': config.get('Entrypoint', [])
        }

    def _analyze_dependencies(self, container) -> Dict[str, Any]:
        """Analyze container dependencies"""
        try:
            # Extract dependency information
            deps = self._extract_dependencies(container)
            
            # Build dependency graph
            dep_graph = self._build_dependency_graph(deps)
            
            return {
                'dependency_count': len(deps),
                'graph_metrics': self._compute_graph_metrics(dep_graph),
                'vulnerability_scan': self._scan_vulnerabilities(deps)
            }
        except Exception as e:
            self.logger.error(f"Dependency analysis failed: {str(e)}")
            return {}

    def _analyze_permissions(self, container) -> Dict[str, Any]:
        """Analyze container permissions"""
        return {
            'capabilities': container.attrs['HostConfig'].get('CapAdd', []),
            'security_opt': container.attrs['HostConfig'].get('SecurityOpt', []),
            'privileged': container.attrs['HostConfig'].get('Privileged', False),
            'user': container.attrs['Config'].get('User', '')
        }

    def _collect_metrics(self, container_id: str) -> Dict[str, float]:
        """Collect real-time container metrics"""
        try:
            container = self.docker_client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            return {
                'cpu_usage': self._calculate_cpu_usage(stats),
                'memory_usage': self._calculate_memory_usage(stats),
                'network_io': self._calculate_network_io(stats),
                'block_io': self._calculate_block_io(stats)
            }
        except Exception as e:
            self.logger.error(f"Metric collection failed: {str(e)}")
            return {}

    def _process_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Process raw metrics"""
        try:
            processed = {}
            for key, value in metrics.items():
                # Apply exponential moving average
                if key in self.feature_buffer:
                    alpha = self.config.get('smoothing_factor', 0.2)
                    processed[key] = alpha * value + (1 - alpha) * self.feature_buffer[-1][key]
                else:
                    processed[key] = value
                    
            # Update buffer
            self.feature_buffer.append(processed)
            if len(self.feature_buffer) > self.window_size:
                self.feature_buffer.pop(0)
                
            return processed
            
        except Exception as e:
            self.logger.error(f"Metric processing failed: {str(e)}")
            return metrics

    def _process_time_series(self, metrics_buffer: List[Dict]) -> Dict[str, Any]:
        """Process time series features"""
        try:
            # Convert to numpy arrays
            time_series = {
                key: np.array([m[key] for m in metrics_buffer])
                for key in metrics_buffer[0].keys()
            }
            
            # Fit ARIMA models
            arima_features = {}
            for key, series in time_series.items():
                model = sm.tsa.ARIMA(series, order=(1,1,1))
                results = model.fit()
                arima_features[key] = {
                    'coefficients': results.params.tolist(),
                    'aic': results.aic
                }
                
            return {
                'arima': arima_features,
                'trends': self._extract_trends(time_series)
            }
            
        except Exception as e:
            self.logger.error(f"Time series processing failed: {str(e)}")
            return {}

    def _compute_statistical_features(self, 
                                    metrics_buffer: List[Dict]) -> Dict[str, float]:
        """Compute statistical features"""
        try:
            stats = {}
            for key in metrics_buffer[0].keys():
                values = [m[key] for m in metrics_buffer]
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'max': np.max(values),
                    'min': np.min(values),
                    'percentiles': np.percentile(values, [25, 50, 75]).tolist()
                }
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistical feature computation failed: {str(e)}")
            return {}

    def _compute_frequency_features(self, 
                                  metrics_buffer: List[Dict]) -> Dict[str, Any]:
        """Compute frequency domain features"""
        try:
            freq_features = {}
            for key in metrics_buffer[0].keys():
                values = [m[key] for m in metrics_buffer]
                
                # Compute FFT
                fft_values = fft(values)
                frequencies = np.fft.fftfreq(len(values), d=self.sampling_rate)
                
                # Extract dominant frequencies
                dominant_freq_idx = np.argsort(np.abs(fft_values))[-3:]
                
                freq_features[key] = {
                    'dominant_frequencies': frequencies[dominant_freq_idx].tolist(),
                    'frequency_magnitudes': np.abs(fft_values[dominant_freq_idx]).tolist()
                }
                
            return freq_features
            
        except Exception as e:
            self.logger.error(f"Frequency feature computation failed: {str(e)}")
            return {}