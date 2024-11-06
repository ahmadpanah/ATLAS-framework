
import docker
import psutil
import numpy as np
from typing import Dict, List, Optional
import threading
import time
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerAttributeAnalyzer:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.container_metrics = defaultdict(dict)
        self.monitoring_thread = None
        self.is_monitoring = False
        self.monitoring_interval = 1  # seconds
        
    def start_monitoring(self):
        """Start continuous container monitoring"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitor_containers)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop container monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            
    def _monitor_containers(self):
        """Continuous container monitoring loop"""
        while self.is_monitoring:
            try:
                containers = self.docker_client.containers.list()
                for container in containers:
                    stats = container.stats(stream=False)
                    self._process_container_stats(container.id, stats)
            except Exception as e:
                logger.error(f"Error monitoring containers: {e}")
            time.sleep(self.monitoring_interval)
            
    def _process_container_stats(self, container_id: str, stats: Dict):
        """Process and store container statistics"""
        try:
            cpu_stats = stats['cpu_stats']
            mem_stats = stats['memory_stats']
            net_stats = stats['networks'] if 'networks' in stats else {}
            
            # Calculate CPU usage percentage
            cpu_delta = cpu_stats['cpu_usage']['total_usage'] - \
                       cpu_stats['cpu_usage'].get('usage_in_kernelmode', 0)
            system_delta = cpu_stats['system_cpu_usage'] - \
                          cpu_stats.get('online_cpus', 1)
            cpu_usage = (cpu_delta / system_delta) * 100.0
            
            # Calculate memory usage
            mem_usage = mem_stats.get('usage', 0)
            mem_limit = mem_stats.get('limit', 1)
            mem_percent = (mem_usage / mem_limit) * 100.0
            
            # Network stats
            net_in = sum(interface['rx_bytes'] for interface in net_stats.values())
            net_out = sum(interface['tx_bytes'] for interface in net_stats.values())
            
            self.container_metrics[container_id].update({
                'timestamp': time.time(),
                'cpu_usage': cpu_usage,
                'memory_usage': mem_percent,
                'network_in': net_in,
                'network_out': net_out
            })
            
        except Exception as e:
            logger.error(f"Error processing container stats: {e}")
            
    def analyze_container(self, container_id: str) -> Dict:
        """Analyze container attributes and security characteristics"""
        try:
            container = self.docker_client.containers.get(container_id)
            inspect_data = container.attrs
            
            # Static analysis
            static_attrs = self._analyze_static_attributes(inspect_data)
            
            # Dynamic analysis
            dynamic_attrs = self._analyze_dynamic_attributes(container_id)
            
            # Security analysis
            security_score = self._calculate_security_score(static_attrs, dynamic_attrs)
            
            return {
                'container_id': container_id,
                'static_attributes': static_attrs,
                'dynamic_attributes': dynamic_attrs,
                'security_score': security_score,
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing container {container_id}: {e}")
            return None
            
    def _analyze_static_attributes(self, inspect_data: Dict) -> Dict:
        """Analyze static container attributes"""
        return {
            'image': inspect_data['Config']['Image'],
            'exposed_ports': list(inspect_data['Config'].get('ExposedPorts', {}).keys()),
            'volumes': list(inspect_data['Config'].get('Volumes', {}).keys()),
            'environment': inspect_data['Config'].get('Env', []),
            'entrypoint': inspect_data['Config'].get('Entrypoint', []),
            'cmd': inspect_data['Config'].get('Cmd', []),
            'labels': inspect_data['Config'].get('Labels', {}),
            'network_mode': inspect_data['HostConfig'].get('NetworkMode', ''),
            'privileged': inspect_data['HostConfig'].get('Privileged', False),
            'security_opts': inspect_data['HostConfig'].get('SecurityOpt', [])
        }
        
    def _analyze_dynamic_attributes(self, container_id: str) -> Dict:
        """Analyze dynamic container attributes"""
        metrics = self.container_metrics.get(container_id, {})
        if not metrics:
            return {}
            
        return {
            'cpu_usage_mean': np.mean([m['cpu_usage'] for m in metrics.values()]),
            'memory_usage_mean': np.mean([m['memory_usage'] for m in metrics.values()]),
            'network_in_rate': self._calculate_rate(metrics, 'network_in'),
            'network_out_rate': self._calculate_rate(metrics, 'network_out'),
            'cpu_usage_std': np.std([m['cpu_usage'] for m in metrics.values()]),
            'memory_usage_std': np.std([m['memory_usage'] for m in metrics.values()])
        }
        
    def _calculate_rate(self, metrics: Dict, key: str) -> float:
        """Calculate rate of change for a metric"""
        values = [m[key] for m in metrics.values()]
        times = [m['timestamp'] for m in metrics.values()]
        if len(values) < 2:
            return 0.0
        return (values[-1] - values[0]) / (times[-1] - times[0])
        
    def _calculate_security_score(self, static_attrs: Dict, dynamic_attrs: Dict) -> float:
        """Calculate security score based on container attributes"""
        score = 100.0
        
        # Static attribute penalties
        if static_attrs.get('privileged', False):
            score -= 30.0
            
        if not static_attrs.get('security_opts'):
            score -= 10.0
            
        if 'host' in static_attrs.get('network_mode', ''):
            score -= 20.0
            
        # Dynamic attribute penalties
        cpu_usage = dynamic_attrs.get('cpu_usage_mean', 0)
        if cpu_usage > 90:
            score -= 15.0
            
        memory_usage = dynamic_attrs.get('memory_usage_mean', 0)
        if memory_usage > 90:
            score -= 15.0
            
        # Network activity analysis
        network_in_rate = dynamic_attrs.get('network_in_rate', 0)
        network_out_rate = dynamic_attrs.get('network_out_rate', 0)
        if network_in_rate > 1e8 or network_out_rate > 1e8:  # 100MB/s threshold
            score -= 10.0
            
        return max(0.0, min(100.0, score))

    def get_container_security_profile(self, container_id: str) -> Dict:
        """Generate comprehensive security profile for a container"""
        analysis = self.analyze_container(container_id)
        if not analysis:
            return None
            
        security_profile = {
            'container_id': container_id,
            'security_score': analysis['security_score'],
            'risk_factors': self._identify_risk_factors(analysis),
            'recommendations': self._generate_recommendations(analysis),
            'security_classification': self._classify_security_level(analysis['security_score']),
            'timestamp': time.time()
        }
        
        return security_profile
        
    def _identify_risk_factors(self, analysis: Dict) -> List[str]:
        """Identify security risk factors"""
        risk_factors = []
        static_attrs = analysis['static_attributes']
        dynamic_attrs = analysis['dynamic_attributes']
        
        if static_attrs.get('privileged'):
            risk_factors.append('Container running in privileged mode')
            
        if 'host' in static_attrs.get('network_mode', ''):
            risk_factors.append('Container using host network mode')
            
        if not static_attrs.get('security_opts'):
            risk_factors.append('No security options configured')
            
        cpu_usage = dynamic_attrs.get('cpu_usage_mean', 0)
        if cpu_usage > 90:
            risk_factors.append(f'High CPU usage: {cpu_usage:.2f}%')
            
        memory_usage = dynamic_attrs.get('memory_usage_mean', 0)
        if memory_usage > 90:
            risk_factors.append(f'High memory usage: {memory_usage:.2f}%')
            
        return risk_factors
        
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        static_attrs = analysis['static_attributes']
        
        if static_attrs.get('privileged'):
            recommendations.append('Remove privileged mode and use specific capabilities')
            
        if 'host' in static_attrs.get('network_mode', ''):
            recommendations.append('Use custom network instead of host network mode')
            
        if not static_attrs.get('security_opts'):
            recommendations.append('Configure security options (e.g., seccomp, apparmor)')
            
        if not static_attrs.get('network_mode').startswith('container:'):
            recommendations.append('Consider using container network isolation')
            
        return recommendations
        
    def _classify_security_level(self, security_score: float) -> str:
        """Classify security level based on score"""
        if security_score >= 90:
            return 'HIGH'
        elif security_score >= 70:
            return 'MEDIUM'
        else:
            return 'LOW'

# Example usage:
if __name__ == "__main__":
    analyzer = ContainerAttributeAnalyzer()
    analyzer.start_monitoring()
    
    try:
        # Analyze all running containers
        containers = analyzer.docker_client.containers.list()
        for container in containers:
            profile = analyzer.get_container_security_profile(container.id)
            print(f"\nSecurity Profile for Container {container.id[:12]}:")
            print(f"Security Score: {profile['security_score']:.2f}")
            print(f"Security Level: {profile['security_classification']}")
            print("\nRisk Factors:")
            for risk in profile['risk_factors']:
                print(f"- {risk}")
            print("\nRecommendations:")
            for rec in profile['recommendations']:
                print(f"- {rec}")
                
    finally:
        analyzer.stop_monitoring()