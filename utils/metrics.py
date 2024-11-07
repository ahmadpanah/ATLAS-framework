
import numpy as np
from typing import Dict, List, Any, Union
import logging
from datetime import datetime, timedelta
from collections import deque

class MetricsCalculator:
    """Utility class for metrics calculations"""
    
    @staticmethod
    def calculate_moving_average(values: List[float], 
                               window: int = 10) -> float:
        """Calculate moving average"""
        if not values:
            return 0.0
        window = min(window, len(values))
        return np.mean(values[-window:])

    @staticmethod
    def calculate_exponential_average(values: List[float], 
                                    alpha: float = 0.2) -> float:
        """Calculate exponential moving average"""
        if not values:
            return 0.0
            
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    @staticmethod
    def calculate_percentile(values: List[float], 
                           percentile: float) -> float:
        """Calculate percentile value"""
        if not values:
            return 0.0
        return float(np.percentile(values, percentile))

    @staticmethod
    def calculate_rate(values: List[float], 
                      time_window: float) -> float:
        """Calculate rate of change"""
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / time_window

class MetricsAggregator:
    """Utility class for metrics aggregation"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_buffer = {}
        self.timestamps = {}

    def add_metric(self, name: str, value: float):
        """Add metric value to buffer"""
        if name not in self.metrics_buffer:
            self.metrics_buffer[name] = deque(maxlen=self.window_size)
            self.timestamps[name] = deque(maxlen=self.window_size)
            
        self.metrics_buffer[name].append(value)
        self.timestamps[name].append(datetime.now())

    def get_statistics(self, name: str) -> Dict[str, float]:
        """Get statistical metrics"""
        if name not in self.metrics_buffer:
            return {}
            
        values = list(self.metrics_buffer[name])
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'last_value': values[-1] if values else 0.0
        }

    def get_trend(self, name: str) -> Dict[str, float]:
        """Get trend metrics"""
        if name not in self.metrics_buffer:
            return {}
            
        values = list(self.metrics_buffer[name])
        times = list(self.timestamps[name])
        
        if len(values) < 2:
            return {'slope': 0.0, 'r_squared': 0.0}
            
        x = np.array([(t - times[0]).total_seconds() 
                      for t in times]).reshape(-1, 1)
        y = np.array(values)
        
        # Linear regression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(x, y)
        
        return {
            'slope': float(model.coef_[0]),
            'r_squared': float(model.score(x, y))
        }

class PerformanceMetricsCalculator:
    """Calculator for performance-specific metrics"""
    
    @staticmethod
    def calculate_latency_score(latency: float, 
                              threshold: float = 100.0) -> float:
        """Calculate latency score"""
        return max(0.0, 1.0 - (latency / threshold))

    @staticmethod
    def calculate_throughput_score(throughput: float,
                                 max_throughput: float) -> float:
        """Calculate throughput score"""
        return min(1.0, throughput / max_throughput)

    @staticmethod
    def calculate_resource_efficiency(used: float,
                                   allocated: float) -> float:
        """Calculate resource efficiency"""
        if allocated == 0:
            return 0.0
        return min(1.0, used / allocated)

class SecurityMetricsCalculator:
    """Calculator for security-specific metrics"""
    
    @staticmethod
    def calculate_encryption_strength(key_size: int,
                                   algorithm: str) -> float:
        """Calculate encryption strength score"""
        base_score = key_size / 256.0
        algorithm_weights = {
            'AES-GCM': 1.0,
            'ChaCha20-Poly1305': 0.95,
            'AES-CBC': 0.9
        }
        return base_score * algorithm_weights.get(algorithm, 0.8)

    @staticmethod
    def calculate_vulnerability_score(vulnerabilities: List[Dict]) -> float:
        """Calculate vulnerability score"""
        if not vulnerabilities:
            return 1.0
            
        severity_weights = {
            'CRITICAL': 1.0,
            'HIGH': 0.8,
            'MEDIUM': 0.5,
            'LOW': 0.2
        }
        
        weighted_sum = sum(
            severity_weights.get(v['severity'], 0.0) 
            for v in vulnerabilities
        )
        return max(0.0, 1.0 - (weighted_sum / len(vulnerabilities)))

    @staticmethod
    def calculate_compliance_score(policies: List[Dict],
                                checks: List[Dict]) -> float:
        """Calculate compliance score"""
        if not checks:
            return 0.0
            
        passed_checks = sum(1 for check in checks if check.get('passed'))
        return passed_checks / len(checks)

class MetricsFormatter:
    """Utility class for metrics formatting"""
    
    @staticmethod
    def format_bytes(bytes_value: float) -> str:
        """Format byte values"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.2f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.2f}PB"

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration values"""
        if seconds < 1:
            return f"{seconds*1000:.2f}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            return f"{seconds/60:.2f}m"
        else:
            return f"{seconds/3600:.2f}h"

    @staticmethod
    def format_percentage(value: float) -> str:
        """Format percentage values"""
        return f"{value*100:.2f}%"