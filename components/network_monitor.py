import time
import threading
import subprocess
import statistics
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import psutil
import numpy as np
from collections import deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkState(Enum):
    OPTIMAL = "OPTIMAL"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNSTABLE = "UNSTABLE"
    FAILED = "FAILED"

@dataclass
class NetworkMetrics:
    bandwidth: float  # Mbps
    latency: float   # ms
    jitter: float    # ms
    packet_loss: float  # percentage
    link_quality: float  # percentage

class NetworkConditionMonitor:
    def __init__(self, target_hosts: List[str], monitoring_interval: float = 1.0):
        self.target_hosts = target_hosts
        self.monitoring_interval = monitoring_interval
        self.metrics_history = {host: deque(maxlen=100) for host in target_hosts}
        self.current_state = {host: NetworkState.OPTIMAL for host in target_hosts}
        self._stop_flag = threading.Event()
        self.monitoring_thread = None
        
        # Thresholds based on paper specifications
        self.thresholds = {
            'bandwidth_min': 100,    # Mbps
            'latency_max': 50,       # ms
            'jitter_max': 10,        # ms
            'packet_loss_max': 0.1,  # percentage
            'link_quality_min': 95   # percentage
        }

    def start_monitoring(self):
        """Start network monitoring"""
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self._stop_flag.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            logger.info("Network monitoring started")

    def stop_monitoring(self):
        """Stop network monitoring"""
        self._stop_flag.set()
        if self.monitoring_thread:
            self.monitoring_thread.join()
            logger.info("Network monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_flag.is_set():
            for host in self.target_hosts:
                try:
                    metrics = self._collect_metrics(host)
                    self.metrics_history[host].append(metrics)
                    self._update_network_state(host)
                except Exception as e:
                    logger.error(f"Error monitoring host {host}: {e}")
            time.sleep(self.monitoring_interval)

    def _collect_metrics(self, host: str) -> NetworkMetrics:
        """Collect network metrics for a target host"""
        bandwidth = self._measure_bandwidth(host)
        latency, jitter = self._measure_latency_jitter(host)
        packet_loss = self._measure_packet_loss(host)
        link_quality = self._calculate_link_quality(bandwidth, latency, packet_loss)

        return NetworkMetrics(
            bandwidth=bandwidth,
            latency=latency,
            jitter=jitter,
            packet_loss=packet_loss,
            link_quality=link_quality
        )

    def _measure_bandwidth(self, host: str) -> float:
        """Measure available bandwidth to target host"""
        try:
            # Using iperf3 for bandwidth measurement
            cmd = f"iperf3 -c {host} -t 1 -J"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return float(data['end']['streams'][0]['receiver']['bits_per_second']) / 1e6
        except Exception as e:
            logger.error(f"Bandwidth measurement error: {e}")
        
        # Fallback to basic estimation
        net_io = psutil.net_io_counters()
        time.sleep(1)
        net_io_after = psutil.net_io_counters()
        bytes_sent = net_io_after.bytes_sent - net_io.bytes_sent
        bytes_recv = net_io_after.bytes_recv - net_io.bytes_recv
        return (bytes_sent + bytes_recv) * 8 / 1e6  # Convert to Mbps

    def _measure_latency_jitter(self, host: str) -> tuple[float, float]:
        """Measure network latency and jitter"""
        latencies = []
        try:
            for _ in range(10):
                cmd = f"ping -c 1 {host}"
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    time_str = result.stdout.split('time=')[-1].split()[0]
                    latencies.append(float(time_str))
                time.sleep(0.1)
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0
                return avg_latency, jitter
        except Exception as e:
            logger.error(f"Latency measurement error: {e}")
        
        return 999.9, 999.9  # Error case

    def _measure_packet_loss(self, host: str) -> float:
        """Measure packet loss rate"""
        try:
            cmd = f"ping -c 100 -i 0.1 {host}"
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode == 0:
                transmitted = int(result.stdout.split()[0])
                received = int(result.stdout.split()[3])
                return ((transmitted - received) / transmitted) * 100
        except Exception as e:
            logger.error(f"Packet loss measurement error: {e}")
        
        return 100.0  # Error case

    def _calculate_link_quality(self, bandwidth: float, latency: float, packet_loss: float) -> float:
        """Calculate overall link quality score"""
        bandwidth_score = min(100, (bandwidth / self.thresholds['bandwidth_min']) * 100)
        latency_score = max(0, 100 - (latency / self.thresholds['latency_max']) * 100)
        loss_score = max(0, 100 - (packet_loss / self.thresholds['packet_loss_max']) * 100)
        
        # Weighted average of scores
        weights = [0.4, 0.4, 0.2]  # Bandwidth, latency, and packet loss weights
        return (bandwidth_score * weights[0] + 
                latency_score * weights[1] + 
                loss_score * weights[2])

    def _update_network_state(self, host: str):
        """Update network state based on current metrics"""
        if not self.metrics_history[host]:
            return

        current_metrics = self.metrics_history[host][-1]
        
        # Calculate metrics stability
        metrics_array = np.array([(m.bandwidth, m.latency, m.packet_loss) 
                                for m in self.metrics_history[host]])
        stability = np.std(metrics_array, axis=0)

        # State determination logic
        if current_metrics.link_quality >= self.thresholds['link_quality_min']:
            new_state = NetworkState.OPTIMAL
        elif current_metrics.bandwidth < self.thresholds['bandwidth_min']:
            new_state = NetworkState.DEGRADED
        elif current_metrics.packet_loss > self.thresholds['packet_loss_max']:
            new_state = NetworkState.CRITICAL
        elif np.any(stability > [10, 5, 0.5]):  # Threshold for bandwidth, latency, loss stability
            new_state = NetworkState.UNSTABLE
        elif current_metrics.link_quality < 50:
            new_state = NetworkState.FAILED
        else:
            new_state = NetworkState.DEGRADED

        if new_state != self.current_state[host]:
            self.current_state[host] = new_state
            logger.info(f"Network state changed for {host}: {new_state.value}")
            self._handle_state_change(host, new_state)

    def _handle_state_change(self, host: str, new_state: NetworkState):
        """Handle network state changes"""
        actions = {
            NetworkState.OPTIMAL: self._handle_optimal_state,
            NetworkState.DEGRADED: self._handle_degraded_state,
            NetworkState.CRITICAL: self._handle_critical_state,
            NetworkState.UNSTABLE: self._handle_unstable_state,
            NetworkState.FAILED: self._handle_failed_state
        }
        
        if new_state in actions:
            actions[new_state](host)

    def _handle_optimal_state(self, host: str):
        """Handle optimal network state"""
        logger.info(f"Network conditions optimal for {host}")
        # Implement optimal state handling logic

    def _handle_degraded_state(self, host: str):
        """Handle degraded network state"""
        logger.warning(f"Network conditions degraded for {host}")
        # Implement degraded state handling logic

    def _handle_critical_state(self, host: str):
        """Handle critical network state"""
        logger.error(f"Critical network conditions for {host}")
        # Implement critical state handling logic

    def _handle_unstable_state(self, host: str):
        """Handle unstable network state"""
        logger.warning(f"Unstable network conditions for {host}")
        # Implement unstable state handling logic

    def _handle_failed_state(self, host: str):
        """Handle failed network state"""
        logger.error(f"Network failure detected for {host}")
        # Implement failed state handling logic

    def get_current_metrics(self, host: str) -> Optional[NetworkMetrics]:
        """Get current network metrics for a host"""
        if host in self.metrics_history and self.metrics_history[host]:
            return self.metrics_history[host][-1]
        return None

    def get_network_state(self, host: str) -> NetworkState:
        """Get current network state for a host"""
        return self.current_state.get(host, NetworkState.FAILED)

    def get_metrics_history(self, host: str) -> List[NetworkMetrics]:
        """Get metrics history for a host"""
        return list(self.metrics_history.get(host, []))

# Example usage
if __name__ == "__main__":
    monitor = NetworkConditionMonitor(["8.8.8.8", "1.1.1.1"])  # Example with Google and Cloudflare DNS
    monitor.start_monitoring()

    try:
        while True:
            for host in monitor.target_hosts:
                metrics = monitor.get_current_metrics(host)
                state = monitor.get_network_state(host)
                if metrics:
                    print(f"\nHost: {host}")
                    print(f"State: {state.value}")
                    print(f"Bandwidth: {metrics.bandwidth:.2f} Mbps")
                    print(f"Latency: {metrics.latency:.2f} ms")
                    print(f"Jitter: {metrics.jitter:.2f} ms")
                    print(f"Packet Loss: {metrics.packet_loss:.2f}%")
                    print(f"Link Quality: {metrics.link_quality:.2f}%")
            time.sleep(5)
    except KeyboardInterrupt:
        monitor.stop_monitoring()