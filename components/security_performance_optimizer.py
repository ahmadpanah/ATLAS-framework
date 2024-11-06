import numpy as np
from typing import Dict, List, Tuple, Optional
import threading
import time
import logging
from dataclasses import dataclass
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class PerformanceLevel(Enum):
    OPTIMAL = "OPTIMAL"
    BALANCED = "BALANCED"
    CONSTRAINED = "CONSTRAINED"

@dataclass
class SecurityConfig:
    encryption_algorithm: str
    key_size: int
    block_size: int
    memory_hardness: int
    iterations: int

@dataclass
class ResourceMetrics:
    cpu_usage: float
    memory_usage: float
    network_bandwidth: float
    disk_io: float
    latency: float

@dataclass
class OptimizationResult:
    security_level: SecurityLevel
    performance_level: PerformanceLevel
    security_config: SecurityConfig
    resource_allocation: Dict[str, float]
    score: float

class SecurityPerformanceOptimizer:
    def __init__(self):
        self.monitoring_interval = 1.0  # seconds
        self.optimization_interval = 5.0  # seconds
        self._stop_flag = threading.Event()
        self.monitoring_thread = None
        self.optimization_thread = None
        self.metrics_history = []
        self.current_optimization: Optional[OptimizationResult] = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_max': 80.0,  # percentage
            'memory_max': 75.0,  # percentage
            'network_max': 90.0,  # percentage
            'latency_max': 50.0,  # ms
            'security_min_score': 85.0  # minimum security score
        }
        
        # Security configurations
        self.security_configs = {
            SecurityLevel.HIGH: SecurityConfig(
                encryption_algorithm="AES-256-GCM",
                key_size=256,
                block_size=128,
                memory_hardness=4,
                iterations=200000
            ),
            SecurityLevel.MEDIUM: SecurityConfig(
                encryption_algorithm="AES-192-GCM",
                key_size=192,
                block_size=128,
                memory_hardness=2,
                iterations=100000
            ),
            SecurityLevel.LOW: SecurityConfig(
                encryption_algorithm="AES-128-GCM",
                key_size=128,
                block_size=128,
                memory_hardness=1,
                iterations=50000
            )
        }

    def start(self):
        """Start the optimizer"""
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self._stop_flag.clear()
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.optimization_thread = threading.Thread(target=self._optimization_loop)
            self.monitoring_thread.daemon = True
            self.optimization_thread.daemon = True
            self.monitoring_thread.start()
            self.optimization_thread.start()
            logger.info("Security-Performance Optimizer started")

    def stop(self):
        """Stop the optimizer"""
        self._stop_flag.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
        if self.optimization_thread:
            self.optimization_thread.join()
        logger.info("Security-Performance Optimizer stopped")

    def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while not self._stop_flag.is_set():
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            time.sleep(self.monitoring_interval)

    def _optimization_loop(self):
        """Continuous optimization loop"""
        while not self._stop_flag.is_set():
            try:
                self._optimize()
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
            time.sleep(self.optimization_interval)

    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics"""
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        
        # Network metrics
        net_io = psutil.net_io_counters()
        time.sleep(0.1)
        net_io_after = psutil.net_io_counters()
        network_bandwidth = ((net_io_after.bytes_sent + net_io_after.bytes_recv) - 
                           (net_io.bytes_sent + net_io.bytes_recv)) / 0.1
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        time.sleep(0.1)
        disk_io_after = psutil.disk_io_counters()
        disk_io_rate = ((disk_io_after.read_bytes + disk_io_after.write_bytes) -
                       (disk_io.read_bytes + disk_io.write_bytes)) / 0.1
        
        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            network_bandwidth=network_bandwidth,
            disk_io=disk_io_rate,
            latency=self._measure_latency()
        )

    def _measure_latency(self) -> float:
        """Measure system latency"""
        start_time = time.time()
        # Perform a small I/O operation
        with open('/dev/null', 'w') as f:
            f.write('test')
        return (time.time() - start_time) * 1000  # Convert to ms

    def _optimize(self):
        """Perform security-performance optimization"""
        if not self.metrics_history:
            return

        # Calculate current resource utilization
        current_metrics = self.metrics_history[-1]
        avg_metrics = self._calculate_average_metrics()
        
        # Determine performance level
        performance_level = self._determine_performance_level(current_metrics)
        
        # Determine security level based on performance constraints
        security_level = self._determine_security_level(performance_level, avg_metrics)
        
        # Calculate resource allocation
        resource_allocation = self._calculate_resource_allocation(
            security_level, performance_level, current_metrics
        )
        
        # Get security configuration
        security_config = self.security_configs[security_level]
        
        # Calculate optimization score
        score = self._calculate_optimization_score(
            security_level, performance_level, current_metrics
        )
        
        # Update current optimization
        self.current_optimization = OptimizationResult(
            security_level=security_level,
            performance_level=performance_level,
            security_config=security_config,
            resource_allocation=resource_allocation,
            score=score
        )
        
        logger.info(f"Optimization updated: Security={security_level.value}, "
                   f"Performance={performance_level.value}, Score={score:.2f}")

    def _calculate_average_metrics(self) -> ResourceMetrics:
        """Calculate average metrics over history"""
        if not self.metrics_history:
            return None
            
        avg_cpu = np.mean([m.cpu_usage for m in self.metrics_history])
        avg_memory = np.mean([m.memory_usage for m in self.metrics_history])
        avg_network = np.mean([m.network_bandwidth for m in self.metrics_history])
        avg_disk = np.mean([m.disk_io for m in self.metrics_history])
        avg_latency = np.mean([m.latency for m in self.metrics_history])
        
        return ResourceMetrics(
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            network_bandwidth=avg_network,
            disk_io=avg_disk,
            latency=avg_latency
        )

    def _determine_performance_level(self, metrics: ResourceMetrics) -> PerformanceLevel:
        """Determine current performance level"""
        if (metrics.cpu_usage > self.thresholds['cpu_max'] or
            metrics.memory_usage > self.thresholds['memory_max']):
            return PerformanceLevel.CONSTRAINED
        elif (metrics.cpu_usage > self.thresholds['cpu_max'] * 0.7 or
              metrics.memory_usage > self.thresholds['memory_max'] * 0.7):
            return PerformanceLevel.BALANCED
        else:
            return PerformanceLevel.OPTIMAL

    def _determine_security_level(self, 
                                performance_level: PerformanceLevel,
                                avg_metrics: ResourceMetrics) -> SecurityLevel:
        """Determine appropriate security level"""
        if performance_level == PerformanceLevel.CONSTRAINED:
            return SecurityLevel.LOW
        elif performance_level == PerformanceLevel.BALANCED:
            return SecurityLevel.MEDIUM
        else:
            return SecurityLevel.HIGH

    def _calculate_resource_allocation(self,
                                    security_level: SecurityLevel,
                                    performance_level: PerformanceLevel,
                                    metrics: ResourceMetrics) -> Dict[str, float]:
        """Calculate optimal resource allocation"""
        base_allocation = {
            'cpu': 20.0,
            'memory': 20.0,
            'network': 20.0,
            'disk': 20.0
        }
        
        # Adjust based on security level
        security_multiplier = {
            SecurityLevel.HIGH: 2.0,
            SecurityLevel.MEDIUM: 1.5,
            SecurityLevel.LOW: 1.0
        }[security_level]
        
        # Adjust based on performance level
        performance_multiplier = {
            PerformanceLevel.OPTIMAL: 1.0,
            PerformanceLevel.BALANCED: 0.8,
            PerformanceLevel.CONSTRAINED: 0.6
        }[performance_level]
        
        return {
            resource: min(base * security_multiplier * performance_multiplier, 100.0)
            for resource, base in base_allocation.items()
        }

    def _calculate_optimization_score(self,
                                   security_level: SecurityLevel,
                                   performance_level: PerformanceLevel,
                                   metrics: ResourceMetrics) -> float:
        """Calculate overall optimization score"""
        security_score = {
            SecurityLevel.HIGH: 1.0,
            SecurityLevel.MEDIUM: 0.7,
            SecurityLevel.LOW: 0.4
        }[security_level]
        
        performance_score = {
            PerformanceLevel.OPTIMAL: 1.0,
            PerformanceLevel.BALANCED: 0.7,
            PerformanceLevel.CONSTRAINED: 0.4
        }[performance_level]
        
        # Resource utilization score
        resource_score = 1.0 - (
            metrics.cpu_usage / 100.0 +
            metrics.memory_usage / 100.0
        ) / 2
        
        # Weighted combination
        weights = {
            'security': 0.4,
            'performance': 0.4,
            'resources': 0.2
        }
        
        return (
            weights['security'] * security_score +
            weights['performance'] * performance_score +
            weights['resources'] * resource_score
        ) * 100.0

    def get_current_optimization(self) -> Optional[OptimizationResult]:
        """Get current optimization result"""
        return self.current_optimization

    def get_metrics_history(self) -> List[ResourceMetrics]:
        """Get metrics history"""
        return self.metrics_history.copy()

# Example usage
if __name__ == "__main__":
    optimizer = SecurityPerformanceOptimizer()
    optimizer.start()

    try:
        while True:
            result = optimizer.get_current_optimization()
            if result:
                print("\nCurrent Optimization Status:")
                print(f"Security Level: {result.security_level.value}")
                print(f"Performance Level: {result.performance_level.value}")
                print(f"Optimization Score: {result.score:.2f}")
                print("\nResource Allocation:")
                for resource, allocation in result.resource_allocation.items():
                    print(f"{resource}: {allocation:.2f}%")
                print("\nSecurity Configuration:")
                print(f"Algorithm: {result.security_config.encryption_algorithm}")
                print(f"Key Size: {result.security_config.key_size}")
                print(f"Memory Hardness: {result.security_config.memory_hardness}")
            time.sleep(5)
    except KeyboardInterrupt:
        optimizer.stop()