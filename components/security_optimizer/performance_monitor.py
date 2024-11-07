
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass
import threading
import time
from collections import deque

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    metrics: Dict[str, float]
    trends: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    timestamp: datetime

class PerformanceMonitor:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Performance Monitor
        
        Args:
            config: Configuration dictionary containing:
                - metrics: List of metrics to monitor
                - window_size: Size of monitoring window
                - update_interval: Metrics update interval
                - anomaly_threshold: Threshold for anomaly detection
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()
        system_config = {
        'collection_interval': self.update_interval,
        'prometheus_port': config.get('prometheus_port', 9090),
        'kubernetes_enabled': config.get('kubernetes_enabled', True),
        'docker_enabled': config.get('docker_enabled', True)
    }
        
        self.system_collector = SystemMetricsCollector(system_config)
        self.system_collector.start_collection()

    def _initialize_components(self):
        """Initialize monitor components"""
        self.metrics_to_monitor = self.config.get('metrics', [])
        self.window_size = self.config.get('window_size', 100)
        self.update_interval = self.config.get('update_interval', 1)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 2.0)
        
        # Initialize metric buffers
        self.metric_buffers = {
            metric: deque(maxlen=self.window_size)
            for metric in self.metrics_to_monitor
        }
        
        # Initialize monitoring thread
        self.is_running = threading.Event()
        self.monitor_thread = None

    def start_monitoring(self):
        """Start performance monitoring"""
        try:
            if not self.is_running.is_set():
                self.is_running.set()
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop
                )
                self.monitor_thread.start()
                self.logger.info("Performance monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            raise

    def stop_monitoring(self):
        """Stop performance monitoring"""
        try:
            if self.is_running.is_set():
                self.is_running.clear()
                if self.monitor_thread:
                    self.monitor_thread.join()
                self.logger.info("Performance monitoring stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {str(e)}")
            raise

    def collect_metrics(self) -> PerformanceMetrics:
        """
        Collect current performance metrics
        
        Returns:
            PerformanceMetrics object containing current metrics
        """
        try:
            # Collect current metrics
            current_metrics = self._collect_current_metrics()
            
            # Update buffers
            self._update_buffers(current_metrics)
            
            # Analyze trends
            trends = self._analyze_trends()
            
            # Detect anomalies
            anomalies = self._detect_anomalies()
            
            return PerformanceMetrics(
                metrics=current_metrics,
                trends=trends,
                anomalies=anomalies,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {str(e)}")
            raise

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running.is_set():
            try:
                # Collect metrics
                metrics = self.collect_metrics()
                
                # Process metrics
                self._process_metrics(metrics)
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                continue

    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current metric values"""
        try:
            metrics = {}
            for metric in self.metrics_to_monitor:
                metrics[metric] = self._get_metric_value(metric)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Current metrics collection failed: {str(e)}")
            return {}

    def _update_buffers(self, metrics: Dict[str, float]):
        """Update metric buffers"""
        try:
            for metric, value in metrics.items():
                if metric in self.metric_buffers:
                    self.metric_buffers[metric].append(value)
                    
        except Exception as e:
            self.logger.error(f"Buffer update failed: {str(e)}")

    def _analyze_trends(self) -> Dict[str, float]:
        """Analyze metric trends"""
        try:
            trends = {}
            for metric, buffer in self.metric_buffers.items():
                if len(buffer) >= 2:
                    # Calculate trend using linear regression
                    x = np.arange(len(buffer))
                    y = np.array(buffer)
                    A = np.vstack([x, np.ones(len(x))]).T
                    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
                    trends[metric] = float(slope)
                else:
                    trends[metric] = 0.0
                    
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {str(e)}")
            return {}

    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect metric anomalies"""
        try:
            anomalies = []
            for metric, buffer in self.metric_buffers.items():
                if len(buffer) >= self.window_size:
                    values = np.array(buffer)
                    mean = np.mean(values)
                    std = np.std(values)
                    
                    # Check for anomalies
                    z_scores = np.abs((values - mean) / std)
                    anomaly_indices = np.where(
                        z_scores > self.anomaly_threshold
                    )[0]
                    
                    if len(anomaly_indices) > 0:
                        anomalies.append({
                            'metric': metric,
                            'indices': anomaly_indices.tolist(),
                            'values': values[anomaly_indices].tolist(),
                            'z_scores': z_scores[anomaly_indices].tolist()
                        })
                        
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            return []

    def _get_metric_value(self, metric: str) -> float:
        """Get current value for metric using SystemMetricsCollector"""
        try:
        # Get metrics from system collector
            metrics = self.system_collector.get_current_metrics()
            return metrics.get(metric, 0.0)
        
        except Exception as e:
            self.logger.error(f"Metric value collection failed: {str(e)}")
        return 0.0

    def _process_metrics(self, metrics: PerformanceMetrics):
        """Process collected metrics"""
        try:
            # Check for anomalies
            if metrics.anomalies:
                self._handle_anomalies(metrics.anomalies)
                
            # Check trends
            self._analyze_performance_trends(metrics.trends)
            
        except Exception as e:
            self.logger.error(f"Metrics processing failed: {str(e)}")

    def _handle_anomalies(self, anomalies: List[Dict[str, Any]]):
        """Handle detected anomalies"""
        try:
            for anomaly in anomalies:
                self.logger.warning(
                    f"Anomaly detected in metric {anomaly['metric']}: "
                    f"z-scores = {anomaly['z_scores']}"
                )
                
        except Exception as e:
            self.logger.error(f"Anomaly handling failed: {str(e)}")

    def _analyze_performance_trends(self, trends: Dict[str, float]):
        """Analyze performance trends"""
        try:
            for metric, trend in trends.items():
                if abs(trend) > self.config.get('trend_threshold', 0.1):
                    self.logger.info(
                        f"Significant trend detected in {metric}: {trend:.3f}"
                    )
                    
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {str(e)}")

    def get_metrics_history(self, 
                          metric: str,
                          window: int = None) -> List[float]:
        """Get historical values for metric"""
        try:
            if metric in self.metric_buffers:
                values = list(self.metric_buffers[metric])
                if window is not None:
                    return values[-window:]
                return values
            return []
            
        except Exception as e:
            self.logger.error(f"History retrieval failed: {str(e)}")
            return []