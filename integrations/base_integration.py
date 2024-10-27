from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import tempfile
import os
import shutil
import json
import hashlib
import socket
from pathlib import Path
from enum import Enum
import subprocess
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerState(Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    RESTARTING = "restarting"
    REMOVING = "removing"
    EXITED = "exited"
    DEAD = "dead"
    MIGRATING = "migrating"

@dataclass
class NetworkConfig:
    interface_name: str
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    gateway: Optional[str] = None
    subnet_mask: Optional[str] = None
    dns_servers: List[str] = None
    exposed_ports: Dict[int, str] = None
    port_bindings: Dict[int, int] = None
    network_mode: str = "bridge"

    def validate(self) -> bool:
        try:
            if self.ip_address:
                socket.inet_aton(self.ip_address)
            if self.gateway:
                socket.inet_aton(self.gateway)
            if self.mac_address and not self._validate_mac(self.mac_address):
                return False
            if self.exposed_ports and not all(isinstance(p, int) for p in self.exposed_ports):
                return False
            return True
        except socket.error:
            return False

    @staticmethod
    def _validate_mac(mac: str) -> bool:
        try:
            parts = mac.split(":")
            return len(parts) == 6 and all(len(p) == 2 and int(p, 16) >= 0 for p in parts)
        except ValueError:
            return False

@dataclass
class StorageConfig:
    volumes: Dict[str, str]  # host_path: container_path
    mounts: List[Dict[str, Any]]
    tmpfs: List[str]
    volume_driver: Optional[str] = None
    storage_opts: Dict[str, str] = None

    def validate(self) -> bool:
        try:
            # Validate volume paths
            for host_path in self.volumes.keys():
                if not os.path.exists(os.path.expanduser(host_path)):
                    return False

            # Validate mounts configuration
            valid_mount_types = ["bind", "volume", "tmpfs", "npipe"]
            for mount in self.mounts:
                if "Type" not in mount or mount["Type"] not in valid_mount_types:
                    return False

            return True
        except Exception:
            return False

@dataclass
class ResourceLimits:
    cpu_count: Optional[float] = None
    cpu_percent: Optional[float] = None
    memory_limit: Optional[int] = None
    memory_swap: Optional[int] = None
    memory_reservation: Optional[int] = None
    kernel_memory: Optional[int] = None
    blkio_weight: Optional[int] = None
    blkio_device_restrictions: List[Dict] = None
    pids_limit: Optional[int] = None

    def validate(self) -> bool:
        try:
            if self.cpu_percent is not None and not 0 <= self.cpu_percent <= 100:
                return False
            if self.memory_limit is not None and self.memory_limit < 0:
                return False
            if self.blkio_weight is not None and not 10 <= self.blkio_weight <= 1000:
                return False
            if self.pids_limit is not None and self.pids_limit < -1:
                return False
            return True
        except Exception:
            return False

@dataclass
class SecurityConfig:
    capabilities_add: List[str] = None
    capabilities_drop: List[str] = None
    privileged: bool = False
    security_opt: List[str] = None
    apparmor_profile: Optional[str] = None
    seccomp_profile: Optional[str] = None
    selinux_label: Optional[str] = None
    no_new_privileges: bool = True
    readonly_rootfs: bool = False
    isolation: str = "default"

    def validate(self) -> bool:
        try:
            valid_isolations = ["default", "process", "hyperv"]
            if self.isolation not in valid_isolations:
                return False

            # Validate capabilities
            valid_capabilities = self._get_valid_capabilities()
            if self.capabilities_add and not all(cap in valid_capabilities for cap in self.capabilities_add):
                return False
            if self.capabilities_drop and not all(cap in valid_capabilities for cap in self.capabilities_drop):
                return False

            # Validate security profiles
            if self.apparmor_profile and not self._validate_apparmor_profile(self.apparmor_profile):
                return False
            if self.seccomp_profile and not self._validate_seccomp_profile(self.seccomp_profile):
                return False

            return True
        except Exception:
            return False

    @staticmethod
    def _get_valid_capabilities() -> List[str]:
        try:
            # Get list of valid capabilities from system
            result = subprocess.run(
                ["capsh", "--print"],
                capture_output=True,
                text=True,
                check=True
            )
            return [
                cap.strip()
                for cap in result.stdout.split("\n")
                if cap.startswith("cap_")
            ]
        except subprocess.SubprocessError:
            # Fallback to common capabilities if capsh not available
            return [
                "cap_audit_write",
                "cap_chown",
                "cap_dac_override",
                "cap_fowner",
                "cap_fsetid",
                "cap_kill",
                "cap_mknod",
                "cap_net_bind_service",
                "cap_net_raw",
                "cap_setfcap",
                "cap_setgid",
                "cap_setpcap",
                "cap_setuid",
                "cap_sys_chroot"
            ]

    @staticmethod
    def _validate_apparmor_profile(profile: str) -> bool:
        try:
            # Check if AppArmor is available
            if not os.path.exists("/sys/kernel/security/apparmor"):
                return False

            # Check if profile exists
            profile_path = f"/etc/apparmor.d/{profile}"
            if not os.path.exists(profile_path):
                return False

            # Validate profile syntax
            result = subprocess.run(
                ["apparmor_parser", "-Q", profile_path],
                capture_output=True,
                check=False
            )
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def _validate_seccomp_profile(profile: str) -> bool:
        try:
            if isinstance(profile, str):
                if not os.path.exists(profile):
                    return False
                with open(profile, 'r') as f:
                    profile_data = json.load(f)
            else:
                profile_data = profile

            required_fields = ["defaultAction", "architectures", "syscalls"]
            return all(field in profile_data for field in required_fields)
        except Exception:
            return False

@dataclass
class ContainerCheckpoint:
    checkpoint_id: str
    creation_time: datetime
    memory_state: bytes
    filesystem_state: bytes
    network_state: Dict
    metadata: Dict
    status: str

    def validate(self) -> bool:
        try:
            if not self.checkpoint_id or not isinstance(self.checkpoint_id, str):
                return False
            if not self.memory_state or not isinstance(self.memory_state, bytes):
                return False
            if not self.filesystem_state or not isinstance(self.filesystem_state, bytes):
                return False
            if not isinstance(self.network_state, dict):
                return False
            if not isinstance(self.metadata, dict):
                return False
            return True
        except Exception:
            return False

    def get_checksum(self) -> str:
        """Calculate checksum of checkpoint data"""
        hasher = hashlib.sha256()
        hasher.update(self.memory_state)
        hasher.update(self.filesystem_state)
        hasher.update(json.dumps(self.network_state).encode())
        hasher.update(json.dumps(self.metadata).encode())
        return hasher.hexdigest()
    
class BaseContainerIntegration:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self._setup_runtime_environment()

    def _setup_runtime_environment(self):
        """Setup runtime environment and verify system requirements"""
        try:
            # Verify system capabilities
            self._verify_system_capabilities()
            
            # Setup logging directory
            self.log_dir = os.path.join(self.temp_dir, 'logs')
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Setup temporary storage for container operations
            self.container_temp = os.path.join(self.temp_dir, 'containers')
            os.makedirs(self.container_temp, exist_ok=True)
            
            # Initialize runtime statistics
            self.runtime_stats = {
                'operations': {},
                'errors': {},
                'performance_metrics': {}
            }
        except Exception as e:
            self.cleanup()
            raise RuntimeError(f"Failed to setup runtime environment: {str(e)}")

    def _verify_system_capabilities(self):
        """Verify system capabilities and requirements"""
        try:
            # Check available memory
            available_memory = psutil.virtual_memory().available
            if available_memory < 1024 * 1024 * 1024:  # 1GB minimum
                raise RuntimeError("Insufficient system memory")

            # Check available disk space
            disk_usage = psutil.disk_usage(self.temp_dir)
            if disk_usage.free < 10 * 1024 * 1024 * 1024:  # 10GB minimum
                raise RuntimeError("Insufficient disk space")

            # Check CPU resources
            if psutil.cpu_count() < 2:
                raise RuntimeError("Insufficient CPU resources")

            # Check system capabilities
            self._check_system_permissions()

        except Exception as e:
            raise RuntimeError(f"System capability verification failed: {str(e)}")

    def _check_system_permissions(self):
        """Check necessary system permissions"""
        try:
            # Check if running with sufficient privileges
            if os.geteuid() != 0:
                raise RuntimeError("Must be run with root privileges")

            # Test file system permissions
            test_file = os.path.join(self.temp_dir, 'permission_test')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except (IOError, OSError) as e:
                raise RuntimeError(f"Insufficient filesystem permissions: {str(e)}")

            # Check network capabilities
            sock = socket.socket(socket.AF_INET, socket.SOCK_RAW, socket.IPPROTO_ICMP)
            sock.close()

        except Exception as e:
            raise RuntimeError(f"Permission check failed: {str(e)}")

    def create_container(self,
                        image: str,
                        name: str,
                        network_config: NetworkConfig,
                        storage_config: StorageConfig,
                        resource_limits: ResourceLimits,
                        security_config: SecurityConfig,
                        **kwargs) -> str:
        """Create a new container with specified configurations"""
        try:
            # Validate all configurations
            if not all([
                network_config.validate(),
                storage_config.validate(),
                resource_limits.validate(),
                security_config.validate()
            ]):
                raise ValueError("Invalid configuration parameters")

            # Create unique container ID
            container_id = self._generate_container_id()

            # Prepare container directory
            container_dir = os.path.join(self.container_temp, container_id)
            os.makedirs(container_dir)

            try:
                # Create container metadata
                metadata = {
                    'id': container_id,
                    'name': name,
                    'image': image,
                    'created': datetime.utcnow().isoformat(),
                    'network_config': network_config.__dict__,
                    'storage_config': storage_config.__dict__,
                    'resource_limits': resource_limits.__dict__,
                    'security_config': security_config.__dict__,
                    'status': 'created',
                    'additional_params': kwargs
                }

                # Save metadata
                with open(os.path.join(container_dir, 'metadata.json'), 'w') as f:
                    json.dump(metadata, f)

                # Setup networking
                self._setup_container_network(container_id, network_config)

                # Setup storage
                self._setup_container_storage(container_id, storage_config)

                # Apply resource limits
                self._apply_resource_limits(container_id, resource_limits)

                # Apply security configuration
                self._apply_security_config(container_id, security_config)

                # Update runtime statistics
                self._update_stats('create_container', True)

                return container_id

            except Exception as e:
                # Cleanup on failure
                shutil.rmtree(container_dir, ignore_errors=True)
                raise

        except Exception as e:
            self._update_stats('create_container', False)
            raise RuntimeError(f"Container creation failed: {str(e)}")

    def _generate_container_id(self) -> str:
        """Generate unique container ID"""
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        random_suffix = os.urandom(4).hex()
        return f"container_{timestamp}_{random_suffix}"

    def _setup_container_network(self,
                               container_id: str,
                               network_config: NetworkConfig):
        """Setup container network configuration"""
        try:
            network_dir = os.path.join(self.container_temp, container_id, 'network')
            os.makedirs(network_dir)

            # Create network namespace
            namespace_path = os.path.join(network_dir, 'netns')
            subprocess.run(['ip', 'netns', 'add', container_id], check=True)

            # Create virtual interface pair
            veth_host = f"veth_{container_id[:8]}"
            veth_container = f"eth0"
            subprocess.run(
                ['ip', 'link', 'add', veth_host, 'type', 'veth',
                 'peer', 'name', veth_container],
                check=True
            )

            # Move container interface to namespace
            subprocess.run(
                ['ip', 'link', 'set', veth_container,
                 'netns', container_id],
                check=True
            )

            # Configure container interface
            if network_config.ip_address:
                subprocess.run(
                    ['ip', 'netns', 'exec', container_id,
                     'ip', 'addr', 'add',
                     f"{network_config.ip_address}/{network_config.subnet_mask}",
                     'dev', veth_container],
                    check=True
                )

            # Configure MAC address if specified
            if network_config.mac_address:
                subprocess.run(
                    ['ip', 'netns', 'exec', container_id,
                     'ip', 'link', 'set', veth_container,
                     'address', network_config.mac_address],
                    check=True
                )

            # Setup DNS configuration
            if network_config.dns_servers:
                resolv_conf = os.path.join(network_dir, 'resolv.conf')
                with open(resolv_conf, 'w') as f:
                    for dns in network_config.dns_servers:
                        f.write(f"nameserver {dns}\n")

            # Save network configuration
            with open(os.path.join(network_dir, 'config.json'), 'w') as f:
                json.dump(network_config.__dict__, f)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Network setup failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Network configuration failed: {str(e)}")

    def _setup_container_storage(self,
                               container_id: str,
                               storage_config: StorageConfig):
        """Setup container storage configuration"""
        try:
            storage_dir = os.path.join(self.container_temp, container_id, 'storage')
            os.makedirs(storage_dir)

            # Setup volume mounts
            for host_path, container_path in storage_config.volumes.items():
                # Create mount point
                mount_point = os.path.join(storage_dir, 'mounts',
                                         hashlib.sha256(container_path.encode()).hexdigest())
                os.makedirs(mount_point, exist_ok=True)

                # Bind mount
                subprocess.run(
                    ['mount', '--bind', host_path, mount_point],
                    check=True
                )

            # Setup tmpfs mounts
            for tmpfs_path in storage_config.tmpfs:
                tmpfs_mount = os.path.join(storage_dir, 'tmpfs',
                                         hashlib.sha256(tmpfs_path.encode()).hexdigest())
                os.makedirs(tmpfs_mount, exist_ok=True)
                subprocess.run(
                    ['mount', '-t', 'tmpfs', 'tmpfs', tmpfs_mount],
                    check=True
                )

            # Save storage configuration
            with open(os.path.join(storage_dir, 'config.json'), 'w') as f:
                json.dump(storage_config.__dict__, f)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Storage mount failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Storage configuration failed: {str(e)}")

    def _apply_resource_limits(self,
                             container_id: str,
                             resource_limits: ResourceLimits):
        """Apply resource limits to container"""
        try:
            cgroup_dir = os.path.join('/sys/fs/cgroup', container_id)
            os.makedirs(cgroup_dir, exist_ok=True)

            # CPU limits
            if resource_limits.cpu_percent is not None:
                cpu_quota = int(resource_limits.cpu_percent * 1000)
                with open(os.path.join(cgroup_dir, 'cpu.cfs_quota_us'), 'w') as f:
                    f.write(str(cpu_quota))

            # Memory limits
            if resource_limits.memory_limit is not None:
                with open(os.path.join(cgroup_dir, 'memory.limit_in_bytes'), 'w') as f:
                    f.write(str(resource_limits.memory_limit))

            # Block I/O limits
            if resource_limits.blkio_weight is not None:
                with open(os.path.join(cgroup_dir, 'blkio.weight'), 'w') as f:
                    f.write(str(resource_limits.blkio_weight))

            # Process limits
            if resource_limits.pids_limit is not None:
                with open(os.path.join(cgroup_dir, 'pids.max'), 'w') as f:
                    f.write(str(resource_limits.pids_limit))

        except Exception as e:
            raise RuntimeError(f"Failed to apply resource limits: {str(e)}")

    def _apply_security_config(self,
                             container_id: str,
                             security_config: SecurityConfig):
        """Apply security configuration to container"""
        try:
            security_dir = os.path.join(self.container_temp, container_id, 'security')
            os.makedirs(security_dir)

            # Apply capabilities
            if security_config.capabilities_add or security_config.capabilities_drop:
                caps_file = os.path.join(security_dir, 'capabilities')
                with open(caps_file, 'w') as f:
                    if security_config.capabilities_add:
                        f.write("Capabilities += " + 
                               " ".join(security_config.capabilities_add) + "\n")
                    if security_config.capabilities_drop:
                        f.write("Capabilities -= " + 
                               " ".join(security_config.capabilities_drop) + "\n")

            # Apply AppArmor profile
            if security_config.apparmor_profile:
                subprocess.run(
                    ['apparmor_parser', '-r', '-W',
                     os.path.join('/etc/apparmor.d',
                                 security_config.apparmor_profile)],
                    check=True
                )

            # Apply Seccomp profile
            if security_config.seccomp_profile:
                seccomp_file = os.path.join(security_dir, 'seccomp.json')
                if isinstance(security_config.seccomp_profile, str):
                    shutil.copy2(security_config.seccomp_profile, seccomp_file)
                else:
                    with open(seccomp_file, 'w') as f:
                        json.dump(security_config.seccomp_profile, f)

            # Save security configuration
            with open(os.path.join(security_dir, 'config.json'), 'w') as f:
                json.dump(security_config.__dict__, f)

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Security profile application failed: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Security configuration failed: {str(e)}")

    def _update_stats(self, operation: str, success: bool):
        """Update runtime statistics"""
        timestamp = datetime.utcnow()
        if operation not in self.runtime_stats['operations']:
            self.runtime_stats['operations'][operation] = {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'last_operation': None
            }

        stats = self.runtime_stats['operations'][operation]
        stats['total'] += 1
        if success:
            stats['successful'] += 1
        else:
            stats['failed'] += 1
        stats['last_operation'] = timestamp

    def cleanup(self):
        """Cleanup resources"""
        try:
            # Remove temporary directory and contents
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)

            # Cleanup network namespaces
            for container_id in os.listdir(self.container_temp):
                try:
                    subprocess.run(['ip', 'netns', 'delete', container_id],
                                 check=False)
                except Exception:
                    pass

            # Remove cgroup directories
            for container_id in os.listdir(self.container_temp):
                cgroup_dir = os.path.join('/sys/fs/cgroup', container_id)
                if os.path.exists(cgroup_dir):
                    shutil.rmtree(cgroup_dir, ignore_errors=True)

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            raise