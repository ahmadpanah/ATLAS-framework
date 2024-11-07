import psutil
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import time
import threading
from collections import deque
import subprocess
import statistics
import re
import socket
from typing import Optional, List, Tuple
import platform

@dataclass
class NetworkMetrics:
    """Network metrics data structure"""
    latency: float
    bandwidth: float
    packet_loss: float
    jitter: float
    throughput: float
    timestamp: datetime

class PingResult:
    def __init__(self, min_rtt: float, avg_rtt: float, max_rtt: float, 
                 mdev: float, packet_loss: float):
        self.min_rtt = min_rtt
        self.avg_rtt = avg_rtt
        self.max_rtt = max_rtt
        self.mdev = mdev
        self.packet_loss = packet_loss

class MetricsCollector:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Network Metrics Collector
        
        Args:
            config: Configuration dictionary containing:
                - sampling_rate: Rate for metric collection
                - window_size: Size of monitoring window
                - interfaces: List of network interfaces to monitor
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()
    
    def _ping_measurement(self, interface: str) -> float:
        """
        Perform ping measurement using ping v3.17.1
        
        Args:
            interface: Network interface to use for ping
            
        Returns:
            float: Measured latency in milliseconds
        """
        try:
            # Get default gateway for the interface
            gateway = self._get_default_gateway(interface)
            if not gateway:
                self.logger.error(f"Could not determine gateway for interface {interface}")
                return float('inf')

            # Perform ping measurement
            ping_result = self._execute_ping(gateway, interface)
            if ping_result:
                return ping_result.avg_rtt
            return float('inf')

        except Exception as e:
            self.logger.error(f"Ping measurement failed: {str(e)}")
            return float('inf')
        
    def _execute_ping(self, target: str, interface: str, 
                     count: int = 5) -> Optional[PingResult]:
        """
        Execute ping command and parse results
        
        Args:
            target: Target IP address or hostname
            interface: Network interface to use
            count: Number of ping packets to send
            
        Returns:
            Optional[PingResult]: Parsed ping results or None if failed
        """
        try:
            # Construct ping command based on OS
            if platform.system().lower() == 'linux':
                cmd = [
                    'ping',
                    '-c', str(count),  # Count
                    '-I', interface,   # Interface
                    '-i', '0.2',       # Interval
                    '-W', '1',         # Timeout
                    target
                ]
            else:  # For other OS (Windows, macOS)
                cmd = [
                    'ping',
                    '-n' if platform.system().lower() == "windows" else '-c',
                    str(count),
                    target
                ]

            # Execute ping command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                return self._parse_ping_output(stdout)
            else:
                self.logger.error(f"Ping failed: {stderr}")
                return None

        except Exception as e:
            self.logger.error(f"Ping execution failed: {str(e)}")
            return None

    def _parse_ping_output(self, output: str) -> Optional[PingResult]:
        """
        Parse ping command output
        
        Args:
            output: String output from ping command
            
        Returns:
            Optional[PingResult]: Parsed results or None if parsing failed
        """
        try:
            # Extract packet loss percentage
            loss_match = re.search(r'(\d+)% packet loss', output)
            packet_loss = float(loss_match.group(1)) / 100 if loss_match else 1.0

            # Extract RTT statistics
            rtt_match = re.search(
                r'rtt min/avg/max/mdev = '
                r'([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)',
                output
            )
            
            if rtt_match:
                return PingResult(
                    min_rtt=float(rtt_match.group(1)),
                    avg_rtt=float(rtt_match.group(2)),
                    max_rtt=float(rtt_match.group(3)),
                    mdev=float(rtt_match.group(4)),
                    packet_loss=packet_loss
                )

            return None

        except Exception as e:
            self.logger.error(f"Ping output parsing failed: {str(e)}")
            return None

    def _get_default_gateway(self, interface: str) -> Optional[str]:
        """
        Get default gateway IP for specified interface
        
        Args:
            interface: Network interface name
            
        Returns:
            Optional[str]: Gateway IP address or None if not found
        """
        try:
            if platform.system().lower() == 'linux':
                # Read route information from /proc/net/route
                with open('/proc/net/route') as f:
                    for line in f:
                        fields = line.strip().split()
                        if fields[0] == interface and int(fields[1], 16) == 0:
                            # Convert hex gateway address to IP
                            gateway = socket.inet_ntoa(
                                bytes.fromhex(fields[2].zfill(8))[::-1]
                            )
                            return gateway
            else:
                # For other OS, implement alternative gateway detection
                # This is a simplified example
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(("8.8.8.8", 80))
                gateway = s.getsockname()[0]
                s.close()
                return gateway

            return None

        except Exception as e:
            self.logger.error(f"Gateway detection failed: {str(e)}")
            return None

    def _calculate_jitter(self, rtt_values: List[float]) -> float:
        """
        Calculate jitter from RTT measurements
        
        Args:
            rtt_values: List of RTT measurements
            
        Returns:
            float: Calculated jitter value
        """
        try:
            if len(rtt_values) < 2:
                return 0.0

            # Calculate differences between consecutive RTT values
            differences = [abs(rtt_values[i] - rtt_values[i-1]) 
                         for i in range(1, len(rtt_values))]
            
            # Calculate mean absolute deviation
            jitter = statistics.mean(differences)
            return jitter

        except Exception as e:
            self.logger.error(f"Jitter calculation failed: {str(e)}")
            return 0.0

    def get_ping_statistics(self, interface: str) -> Dict[str, float]:
        """
        Get comprehensive ping statistics
        
        Args:
            interface: Network interface name
            
        Returns:
            Dict[str, float]: Dictionary containing ping statistics
        """
        try:
            gateway = self._get_default_gateway(interface)
            if not gateway:
                return {
                    'min_rtt': float('inf'),
                    'avg_rtt': float('inf'),
                    'max_rtt': float('inf'),
                    'jitter': 0.0,
                    'packet_loss': 1.0
                }

            result = self._execute_ping(gateway, interface, count=10)
            if result:
                return {
                    'min_rtt': result.min_rtt,
                    'avg_rtt': result.avg_rtt,
                    'max_rtt': result.max_rtt,
                    'jitter': result.mdev,
                    'packet_loss': result.packet_loss
                }

            return {
                'min_rtt': float('inf'),
                'avg_rtt': float('inf'),
                'max_rtt': float('inf'),
                'jitter': 0.0,
                'packet_loss': 1.0
            }

        except Exception as e:
            self.logger.error(f"Failed to get ping statistics: {str(e)}")
            return {
                'min_rtt': float('inf'),
                'avg_rtt': float('inf'),
                'max_rtt': float('inf'),
                'jitter': 0.0,
                'packet_loss': 1.0
            }
        
    def _initialize_components(self):
        """Initialize collector components"""
        self.sampling_rate = self.config.get('sampling_rate', 1.0)
        self.window_size = self.config.get('window_size', 100)
        self.interfaces = self.config.get('interfaces', ['eth0'])
        
        # Initialize buffers
        self.metrics_buffer = {
            interface: deque(maxlen=self.window_size)
            for interface in self.interfaces
        }
        
        # Initialize monitoring thread
        self.is_running = threading.Event()
        self.collection_thread = None

    def start_collection(self):
        """Start metrics collection"""
        try:
            if not self.is_running.is_set():
                self.is_running.set()
                self.collection_thread = threading.Thread(
                    target=self._collection_loop
                )
                self.collection_thread.start()
                self.logger.info("Metrics collection started")
        except Exception as e:
            self.logger.error(f"Failed to start metrics collection: {str(e)}")
            raise

    def stop_collection(self):
        """Stop metrics collection"""
        try:
            if self.is_running.is_set():
                self.is_running.clear()
                if self.collection_thread:
                    self.collection_thread.join()
                self.logger.info("Metrics collection stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop metrics collection: {str(e)}")
            raise

    def _collection_loop(self):
        """Main collection loop"""
        while self.is_running.is_set():
            try:
                for interface in self.interfaces:
                    metrics = self._collect_interface_metrics(interface)
                    self.metrics_buffer[interface].append(metrics)
                
                time.sleep(1.0 / self.sampling_rate)
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {str(e)}")
                continue

    def _collect_interface_metrics(self, interface: str) -> NetworkMetrics:
        """Collect metrics for specific interface"""
        try:
            # Collect raw metrics
            latency = self._measure_latency(interface)
            bandwidth = self._measure_bandwidth(interface)
            packet_loss = self._measure_packet_loss(interface)
            jitter = self._measure_jitter(interface)
            throughput = self._measure_throughput(interface)
            
            # Create metrics object
            metrics = NetworkMetrics(
                latency=latency,
                bandwidth=bandwidth,
                packet_loss=packet_loss,
                jitter=jitter,
                throughput=throughput,
                timestamp=datetime.now()
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Interface metrics collection failed: {str(e)}")
            raise

    def _measure_latency(self, interface: str) -> float:
        """Measure network latency"""
        try:
            # Implement latency measurement logic
            # Could use ping or custom probe packets
            return self._ping_measurement(interface)
        except Exception as e:
            self.logger.error(f"Latency measurement failed: {str(e)}")
            return float('inf')

    def _measure_bandwidth(self, interface: str) -> float:
        """Measure network bandwidth"""
        try:
            # Get interface statistics
            stats = psutil.net_io_counters(pernic=True)[interface]
            current_time = time.time()
            
            # Calculate bandwidth
            if hasattr(self, '_last_bandwidth_check'):
                time_delta = current_time - self._last_bandwidth_check
                bytes_delta = (stats.bytes_sent + stats.bytes_recv -
                             self._last_bytes_total)
                bandwidth = bytes_delta / time_delta
            else:
                bandwidth = 0
                
            # Update last check
            self._last_bandwidth_check = current_time
            self._last_bytes_total = stats.bytes_sent + stats.bytes_recv
            
            return bandwidth
            
        except Exception as e:
            self.logger.error(f"Bandwidth measurement failed: {str(e)}")
            return 0.0

    def _measure_packet_loss(self, interface: str) -> float:
        """Measure packet loss rate"""
        try:
            # Get interface statistics
            stats = psutil.net_io_counters(pernic=True)[interface]
            current_time = time.time()
            
            # Calculate packet loss
            if hasattr(self, '_last_packet_check'):
                packets_sent_delta = (stats.packets_sent - 
                                    self._last_packets_sent)
                packets_recv_delta = (stats.packets_recv - 
                                    self._last_packets_recv)
                
                if packets_sent_delta > 0:
                    loss_rate = 1.0 - (packets_recv_delta / packets_sent_delta)
                else:
                    loss_rate = 0.0
            else:
                loss_rate = 0.0
                
            # Update last check
            self._last_packet_check = current_time
            self._last_packets_sent = stats.packets_sent
            self._last_packets_recv = stats.packets_recv
            
            return loss_rate
            
        except Exception as e:
            self.logger.error(f"Packet loss measurement failed: {str(e)}")
            return 0.0

    def _measure_jitter(self, interface: str) -> float:
        """Measure network jitter"""
        try:
            # Calculate jitter from latency measurements
            if len(self.metrics_buffer[interface]) > 1:
                latencies = [m.latency for m in self.metrics_buffer[interface]]
                differences = np.diff(latencies)
                jitter = np.std(differences)
            else:
                jitter = 0.0
                
            return jitter
            
        except Exception as e:
            self.logger.error(f"Jitter measurement failed: {str(e)}")
            return 0.0

    def _measure_throughput(self, interface: str) -> float:
        """Measure network throughput"""
        try:
            # Get interface statistics
            stats = psutil.net_io_counters(pernic=True)[interface]
            current_time = time.time()
            
            # Calculate throughput
            if hasattr(self, '_last_throughput_check'):
                time_delta = current_time - self._last_throughput_check
                bytes_sent_delta = stats.bytes_sent - self._last_bytes_sent
                bytes_recv_delta = stats.bytes_recv - self._last_bytes_recv
                
                throughput = (bytes_sent_delta + bytes_recv_delta) / time_delta
            else:
                throughput = 0.0
                
            # Update last check
            self._last_throughput_check = current_time
            self._last_bytes_sent = stats.bytes_sent
            self._last_bytes_recv = stats.bytes_recv
            
            return throughput
            
        except Exception as e:
            self.logger.error(f"Throughput measurement failed: {str(e)}")
            return 0.0

    def _ping_measurement(self, interface: str) -> float:
        """Perform ping measurement"""
        try:
            # Implement ping logic here
            # Could use subprocess to call ping command
            # or implement custom ICMP echo request
            return 0.0  # Placeholder
        except Exception as e:
            self.logger.error(f"Ping measurement failed: {str(e)}")
            return float('inf')

    def get_current_metrics(self, interface: str) -> Optional[NetworkMetrics]:
        """Get most recent metrics for interface"""
        try:
            if self.metrics_buffer[interface]:
                return self.metrics_buffer[interface][-1]
            return None
        except Exception as e:
            self.logger.error(f"Failed to get current metrics: {str(e)}")
            return None

    def get_metrics_history(self, 
                          interface: str,
                          window: int = None) -> List[NetworkMetrics]:
        """Get metrics history for interface"""
        try:
            buffer = list(self.metrics_buffer[interface])
            if window is not None:
                return buffer[-window:]
            return buffer
        except Exception as e:
            self.logger.error(f"Failed to get metrics history: {str(e)}")
            return []