import docker
import tarfile
import io
import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import psutil
import tempfile
from .base_integration import BaseContainerIntegration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DockerIntegration(BaseContainerIntegration):
    """Docker-specific implementation of container operations"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.api_client = docker.APIClient()
        self.temp_dir = tempfile.mkdtemp()

    def export_container(self, container_id: str) -> bytes:
        """Export container as tar archive"""
        try:
            container = self.client.containers.get(container_id)
            
            # Create container checkpoint
            checkpoint_data = self.api_client.create_container_checkpoint(
                container.id,
                {
                    'checkpoint': True,
                    'exit': False,
                    'leave-running': True
                }
            )

            # Export container filesystem
            export_data = self.api_client.export(container.id)
            
            # Create tar archive in memory
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
                # Add container metadata
                metadata = {
                    'config': container.attrs,
                    'checkpoint': checkpoint_data,
                    'timestamp': datetime.now().isoformat()
                }
                metadata_bytes = json.dumps(metadata).encode('utf-8')
                metadata_file = io.BytesIO(metadata_bytes)
                
                metadata_info = tarfile.TarInfo('metadata.json')
                metadata_info.size = len(metadata_bytes)
                tar.addfile(metadata_info, metadata_file)
                
                # Add container filesystem
                for chunk in export_data:
                    tar.addfile(tarfile.TarInfo('fs.tar'), io.BytesIO(chunk))

            return tar_buffer.getvalue()

        except Exception as e:
            logger.error(f"Container export failed: {str(e)}")
            raise

    def import_container(self, data: bytes, target_name: str) -> str:
        """Import container from tar archive"""
        try:
            # Create temporary directory for extraction
            temp_extract_dir = os.path.join(self.temp_dir, target_name)
            os.makedirs(temp_extract_dir, exist_ok=True)
            
            # Extract archive
            tar_buffer = io.BytesIO(data)
            with tarfile.open(fileobj=tar_buffer, mode='r:gz') as tar:
                tar.extractall(temp_extract_dir)
            
            # Read metadata
            with open(os.path.join(temp_extract_dir, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
            
            # Create new container
            container_config = metadata['config']
            fs_path = os.path.join(temp_extract_dir, 'fs.tar')
            
            # Import filesystem
            with open(fs_path, 'rb') as f:
                image = self.client.images.load(f.read())[0]
            
            # Create and start container
            container = self.client.containers.create(
                image.id,
                name=target_name,
                **self._prepare_container_config(container_config)
            )
            
            # Restore checkpoint if available
            if 'checkpoint' in metadata:
                self.api_client.restore_container_checkpoint(
                    container.id,
                    metadata['checkpoint']
                )
            
            container.start()
            return container.id

        except Exception as e:
            logger.error(f"Container import failed: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_extract_dir):
                os.rmdir(temp_extract_dir)

    def get_container_metrics(self, container_id: str) -> Dict:
        """Get detailed container metrics"""
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # CPU metrics
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_cpu_delta = stats['cpu_stats']['system_cpu_usage'] - \
                             stats['precpu_stats']['system_cpu_usage']
            number_cpus = len(stats['cpu_stats']['cpu_usage']['percpu_usage'])
            cpu_usage = (cpu_delta / system_cpu_delta) * number_cpus * 100.0
            
            # Memory metrics
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            # Network metrics
            network_stats = stats['networks']['eth0']
            
            # Block I/O metrics
            io_stats = stats['blkio_stats']['io_service_bytes_recursive']
            read_bytes = sum(stat['value'] for stat in io_stats if stat['op'] == 'Read')
            write_bytes = sum(stat['value'] for stat in io_stats if stat['op'] == 'Write')
            
            return {
                'container_id': container_id,
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'usage_percent': cpu_usage,
                    'cores': number_cpus,
                    'throttling_data': stats['cpu_stats'].get('throttling_data', {})
                },
                'memory': {
                    'usage_bytes': memory_usage,
                    'limit_bytes': memory_limit,
                    'usage_percent': memory_percent,
                    'cache': stats['memory_stats'].get('stats', {}).get('cache', 0),
                    'rss': stats['memory_stats'].get('stats', {}).get('rss', 0)
                },
                'network': {
                    'rx_bytes': network_stats['rx_bytes'],
                    'tx_bytes': network_stats['tx_bytes'],
                    'rx_packets': network_stats['rx_packets'],
                    'tx_packets': network_stats['tx_packets'],
                    'rx_errors': network_stats['rx_errors'],
                    'tx_errors': network_stats['tx_errors']
                },
                'io': {
                    'read_bytes': read_bytes,
                    'write_bytes': write_bytes
                }
            }

        except Exception as e:
            logger.error(f"Failed to get container metrics: {str(e)}")
            raise

    def get_container_security_config(self, container_id: str) -> Dict:
        """Get container security configuration"""
        try:
            container = self.client.containers.get(container_id)
            config = container.attrs['HostConfig']
            
            return {
                'container_id': container_id,
                'capabilities': {
                    'added': config.get('CapAdd', []),
                    'dropped': config.get('CapDrop', [])
                },
                'security_opt': config.get('SecurityOpt', []),
                'privileged': config.get('Privileged', False),
                'isolation': config.get('Isolation', ''),
                'user': container.attrs['Config'].get('User', ''),
                'read_only': config.get('ReadonlyRootfs', False),
                'cgroup_parent': config.get('CgroupParent', ''),
                'security_profiles': self._get_security_profiles(container_id)
            }

        except Exception as e:
            logger.error(f"Failed to get security config: {str(e)}")
            raise

    def update_container_security(self, container_id: str, security_config: Dict) -> None:
        """Update container security configuration"""
        try:
            container = self.client.containers.get(container_id)
            
            # Container must be stopped to update security config
            was_running = container.status == 'running'
            if was_running:
                container.stop(timeout=10)
            
            # Update security configuration
            update_config = {
                'CapAdd': security_config['capabilities']['added'],
                'CapDrop': security_config['capabilities']['dropped'],
                'SecurityOpt': security_config['security_opt'],
                'Privileged': security_config['privileged'],
                'ReadonlyRootfs': security_config['read_only']
            }
            
            container.update(**update_config)
            
            # Restart if it was running
            if was_running:
                container.start()

        except Exception as e:
            logger.error(f"Failed to update security config: {str(e)}")
            raise

    def _prepare_container_config(self, original_config: Dict) -> Dict:
        """Prepare container configuration for import"""
        config = {
            'hostname': original_config['Config']['Hostname'],
            'domainname': original_config['Config']['Domainname'],
            'user': original_config['Config']['User'],
            'attach_stdin': False,
            'attach_stdout': False,
            'attach_stderr': False,
            'exposed_ports': original_config['Config']['ExposedPorts'],
            'tty': original_config['Config']['Tty'],
            'open_stdin': original_config['Config']['OpenStdin'],
            'stdin_once': original_config['Config']['StdinOnce'],
            'env': original_config['Config']['Env'],
            'cmd': original_config['Config']['Cmd'],
            'healthcheck': original_config['Config'].get('Healthcheck', None),
            'volumes': original_config['Config']['Volumes'],
            'working_dir': original_config['Config']['WorkingDir'],
            'entrypoint': original_config['Config']['Entrypoint'],
            'labels': original_config['Config']['Labels'],
            'host_config': self._prepare_host_config(original_config['HostConfig'])
        }
        
        return config

    def _prepare_host_config(self, original_host_config: Dict) -> Dict:
        """Prepare host configuration for import"""
        return {
            'binds': original_host_config['Binds'],
            'port_bindings': original_host_config['PortBindings'],
            'restart_policy': original_host_config['RestartPolicy'],
            'auto_remove': original_host_config['AutoRemove'],
            'volume_driver': original_host_config['VolumeDriver'],
            'volumes_from': original_host_config['VolumesFrom'],
            'cap_add': original_host_config['CapAdd'],
            'cap_drop': original_host_config['CapDrop'],
            'dns': original_host_config['Dns'],
            'dns_options': original_host_config['DnsOptions'],
            'dns_search': original_host_config['DnsSearch'],
            'extra_hosts': original_host_config['ExtraHosts'],
            'network_mode': original_host_config['NetworkMode'],
            'security_opt': original_host_config['SecurityOpt'],
            'privileged': original_host_config['Privileged'],
            'readonly_rootfs': original_host_config['ReadonlyRootfs']
        }

    def _get_security_profiles(self, container_id: str) -> Dict:
        """Get detailed security profiles"""
        try:
            # Get AppArmor profile
            apparmor_profile = self.api_client.inspect_container(container_id) \
                             .get('AppArmorProfile', '')
            
            # Get Seccomp profile
            seccomp_profile = self.api_client.inspect_container(container_id) \
                            .get('HostConfig', {}).get('SecurityOpt', [])
            
            return {
                'apparmor': apparmor_profile,
                'seccomp': seccomp_profile
            }

        except Exception as e:
            logger.error(f"Failed to get security profiles: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup temporary resources"""
        try:
            if os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")