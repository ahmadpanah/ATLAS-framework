import logging
from typing import Dict, Optional, List
from enum import Enum
from datetime import datetime
from threading import Lock
import uuid
import time
from dataclasses import asdict

from .federated_learning import FederatedLearningModule
from .container_analyzer import ContainerAttributeAnalyzer
from .network_monitor import NetworkConditionMonitor
from .encryption_engine import AdaptiveEncryptionEngine
from ..integrations import DockerIntegration
from .optimizer import SecurityPerformanceOptimizer
from ..utils.data_structures import (
    ContainerMetrics,
    NetworkMetrics,
    SecurityLevel,
    ContainerAttributes,
    MigrationRequest
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MigrationState(Enum):
    INITIATED = "INITIATED"
    ANALYZING = "ANALYZING"
    PREPARING = "PREPARING"
    ENCRYPTING = "ENCRYPTING"
    TRANSFERRING = "TRANSFERRING"
    DECRYPTING = "DECRYPTING"
    VERIFYING = "VERIFYING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class MigrationController:
    """
    Main controller for the ATLAS framework.
    Coordinates all components to manage secure container migration.
    """
    def __init__(self, container_runtime: str = 'docker'):
        # Initialize all components
        self.flm = FederatedLearningModule()
        self.caa = ContainerAttributeAnalyzer()
        self.ncm = NetworkConditionMonitor()
        self.aee = AdaptiveEncryptionEngine()
        self.spo = SecurityPerformanceOptimizer()

        if container_runtime == 'docker':
            self.container_runtime = DockerIntegration()
        else:
            raise ValueError(f"Unsupported container runtime: {container_runtime}")
        
        # Thread safety
        self.lock = Lock()
        
        # Migration tracking
        self.active_migrations: Dict[str, Dict] = {}
        self.migration_history: Dict[str, Dict] = {}
        
        # Start network monitoring
        self.ncm.start_monitoring()
        
        logger.info("Migration Controller initialized successfully")

    def initiate_migration(self, request: MigrationRequest) -> Dict:
        """
        Initiate a new container migration process
        """
        try:
            migration_id = str(uuid.uuid4())
            
            with self.lock:
                # Initialize migration state
                self.active_migrations[migration_id] = {
                    'id': migration_id,
                    'request': request,
                    'state': MigrationState.INITIATED,
                    'start_time': datetime.now(),
                    'stages': {},
                    'current_stage': None,
                    'progress': 0.0,
                    'errors': []
                }
            
            # Start migration process asynchronously
            self._start_migration_process(migration_id)
            
            return {
                'migration_id': migration_id,
                'status': 'initiated',
                'message': f'Migration process started for container {request.container_id}'
            }
            
        except Exception as e:
            logger.error(f"Failed to initiate migration: {str(e)}")
            raise

    def _start_migration_process(self, migration_id: str):
        """
        Handle the complete migration process
        """
        try:
            # 1. Pre-migration Security Assessment
            if not self._perform_security_assessment(migration_id):
                return

            # 2. Network Condition Assessment
            if not self._assess_network_conditions(migration_id):
                return

            # 3. Resource Optimization
            if not self._optimize_resources(migration_id):
                return

            # 4. Encryption Configuration
            if not self._configure_encryption(migration_id):
                return

            # 5. Container Migration
            if not self._perform_migration(migration_id):
                return

            # 6. Post-migration Verification
            if not self._verify_migration(migration_id):
                return

            # Complete migration
            self._complete_migration(migration_id)

        except Exception as e:
            self._handle_migration_error(migration_id, str(e))

    def _perform_security_assessment(self, migration_id: str) -> bool:
        """
        Perform pre-migration security assessment using FLM and CAA
        """
        try:
            migration = self._get_migration(migration_id)
            self._update_migration_state(migration_id, MigrationState.ANALYZING)

            # Get container attributes
            container_attrs = self._get_container_attributes(
                migration['request'].container_id
            )

            # Analyze container security
            security_analysis = self.caa.analyze_container(container_attrs)

            # Get threat assessment from FLM
            threat_assessment = self.flm.assess_security_requirements(
                asdict(container_attrs)
            )

            # Combine assessments
            final_assessment = self._combine_security_assessments(
                security_analysis,
                threat_assessment
            )

            # Update migration state
            migration['security_assessment'] = final_assessment
            migration['progress'] = 0.20

            logger.info(f"Security assessment completed for migration {migration_id}")
            return True

        except Exception as e:
            self._handle_migration_error(
                migration_id,
                f"Security assessment failed: {str(e)}"
            )
            return False

    def _assess_network_conditions(self, migration_id: str) -> bool:
        """
        Assess network conditions using NCM
        """
        try:
            migration = self._get_migration(migration_id)

            # Get current network conditions
            network_metrics = self.ncm.get_current_metrics()
            network_quality = self.ncm.analyze_network_quality()

            # Predict network conditions
            network_prediction = self.ncm.predict_network_conditions()

            # Check if network conditions are suitable
            if network_quality['quality'] == "POOR":
                self._handle_migration_error(
                    migration_id,
                    "Network conditions unsuitable for migration"
                )
                return False

            # Update migration state
            migration['network_assessment'] = {
                'current_metrics': network_metrics,
                'quality': network_quality,
                'prediction': network_prediction
            }
            migration['progress'] = 0.35

            logger.info(f"Network assessment completed for migration {migration_id}")
            return True

        except Exception as e:
            self._handle_migration_error(
                migration_id,
                f"Network assessment failed: {str(e)}"
            )
            return False

    def _optimize_resources(self, migration_id: str) -> bool:
        """
        Optimize resource allocation using SPO
        """
        try:
            migration = self._get_migration(migration_id)
            self._update_migration_state(migration_id, MigrationState.PREPARING)

            # Get container metrics
            container_metrics = self._get_container_metrics(
                migration['request'].container_id
            )

            # Optimize resources
            optimization_result = self.spo.optimize_resources(
                migration['request'].container_id,
                migration['security_assessment']['security_level'],
                migration['network_assessment']['current_metrics'],
                container_metrics
            )

            # Update migration state
            migration['resource_optimization'] = optimization_result
            migration['progress'] = 0.50

            logger.info(f"Resource optimization completed for migration {migration_id}")
            return True

        except Exception as e:
            self._handle_migration_error(
                migration_id,
                f"Resource optimization failed: {str(e)}"
            )
            return False

    def _configure_encryption(self, migration_id: str) -> bool:
        """
        Configure encryption using AEE
        """
        try:
            migration = self._get_migration(migration_id)
            self._update_migration_state(migration_id, MigrationState.ENCRYPTING)

            # Adapt encryption parameters
            encryption_config = self.aee.adapt_encryption(
                migration['request'].container_id,
                migration['security_assessment']['security_level'],
                migration['network_assessment']['current_metrics']
            )

            # Update migration state
            migration['encryption_config'] = encryption_config
            migration['progress'] = 0.65

            logger.info(f"Encryption configured for migration {migration_id}")
            return True

        except Exception as e:
            self._handle_migration_error(
                migration_id,
                f"Encryption configuration failed: {str(e)}"
            )
            return False

    def _perform_migration(self, migration_id: str) -> bool:
        """
        Perform the actual container migration
        """
        try:
            migration = self._get_migration(migration_id)
            self._update_migration_state(migration_id, MigrationState.TRANSFERRING)

            # Get container data
            container_data = self._get_container_data(
                migration['request'].container_id
            )

            # Encrypt container data
            encrypted_data = self.aee.encrypt_data(
                migration['request'].container_id,
                container_data
            )

            # Transfer container
            self._transfer_container(
                migration_id,
                encrypted_data,
                migration['request'].destination_cloud
            )

            # Decrypt at destination
            self._decrypt_container(
                migration_id,
                encrypted_data,
                migration['request'].destination_cloud
            )

            migration['progress'] = 0.85
            return True

        except Exception as e:
            self._handle_migration_error(
                migration_id,
                f"Migration transfer failed: {str(e)}"
            )
            return False

    def _verify_migration(self, migration_id: str) -> bool:
        """
        Verify migration success
        """
        try:
            migration = self._get_migration(migration_id)
            self._update_migration_state(migration_id, MigrationState.VERIFYING)

            # Verify container integrity
            integrity_verified = self._verify_container_integrity(
                migration['request'].container_id,
                migration['request'].destination_cloud
            )

            # Verify security configuration
            security_verified = self._verify_security_configuration(
                migration_id,
                migration['request'].destination_cloud
            )

            # Verify resource allocation
            resources_verified = self._verify_resource_allocation(
                migration_id,
                migration['request'].destination_cloud
            )

            if not all([integrity_verified, security_verified, resources_verified]):
                raise ValueError("Migration verification failed")

            migration['progress'] = 0.95
            return True

        except Exception as e:
            self._handle_migration_error(
                migration_id,
                f"Migration verification failed: {str(e)}"
            )
            return False

    def _complete_migration(self, migration_id: str):
        """
        Complete the migration process
        """
        with self.lock:
            migration = self.active_migrations[migration_id]
            migration['state'] = MigrationState.COMPLETED
            migration['progress'] = 1.0
            migration['completion_time'] = datetime.now()
            
            # Move to history
            self.migration_history[migration_id] = migration
            del self.active_migrations[migration_id]
            
            logger.info(f"Migration {migration_id} completed successfully")

    def _handle_migration_error(self, migration_id: str, error_message: str):
        """
        Handle migration errors
        """
        with self.lock:
            if migration_id in self.active_migrations:
                migration = self.active_migrations[migration_id]
                migration['state'] = MigrationState.FAILED
                migration['errors'].append({
                    'time': datetime.now(),
                    'message': error_message
                })
                
                # Move to history
                self.migration_history[migration_id] = migration
                del self.active_migrations[migration_id]
                
                logger.error(f"Migration {migration_id} failed: {error_message}")

    # Helper methods
    def _get_migration(self, migration_id: str) -> Dict:
        """Get migration state with lock"""
        with self.lock:
            if migration_id not in self.active_migrations:
                raise ValueError(f"No active migration found with ID {migration_id}")
            return self.active_migrations[migration_id]

    def _update_migration_state(self, migration_id: str, state: MigrationState):
        """Update migration state with lock"""
        with self.lock:
            if migration_id in self.active_migrations:
                self.active_migrations[migration_id]['state'] = state
                self.active_migrations[migration_id]['current_stage'] = state.value
                logger.info(f"Migration {migration_id} state updated to {state.value}")

    # Interface methods that need to be implemented based on the container runtime
    def _get_container_attributes(self, container_id: str) -> ContainerAttributes:
        """Get container attributes from runtime"""
        # Implement based on container runtime (Docker, Kubernetes, etc.)
        pass

    def _get_container_metrics(self, container_id: str) -> ContainerMetrics:
        """Get container metrics from runtime"""
        # Implement based on container runtime
        pass

    def _get_container_data(self, container_id: str) -> bytes:
        """Get container data for migration"""
        # Implement based on container runtime
        pass

    def _transfer_container(self, migration_id: str, encrypted_data: Dict, destination: str):
        """Transfer container to destination"""
        # Implement based on infrastructure
        pass

    def _decrypt_container(self, migration_id: str, encrypted_data: Dict, destination: str):
        """Decrypt container at destination"""
        # Implement based on infrastructure
        pass

    def _verify_container_integrity(self, container_id: str, destination: str) -> bool:
        """Verify container integrity after migration"""
        # Implement based on container runtime
        pass

    def _verify_security_configuration(self, migration_id: str, destination: str) -> bool:
        """Verify security configuration after migration"""
        # Implement based on container runtime
        pass

    def _verify_resource_allocation(self, migration_id: str, destination: str) -> bool:
        """Verify resource allocation after migration"""
        # Implement based on container runtime
        pass

    def get_migration_status(self, migration_id: str) -> Optional[Dict]:
        """Get current status of a migration"""
        with self.lock:
            if migration_id in self.active_migrations:
                migration = self.active_migrations[migration_id]
            elif migration_id in self.migration_history:
                migration = self.migration_history[migration_id]
            else:
                return None

            return {
                'migration_id': migration_id,
                'container_id': migration['request'].container_id,
                'state': migration['state'].value,
                'progress': migration['progress'],
                'current_stage': migration['current_stage'],
                'start_time': migration['start_time'],
                'completion_time': migration.get('completion_time'),
                'errors': migration['errors']
            }

    def get_active_migrations(self) -> List[str]:
        """Get list of active migration IDs"""
        with self.lock:
            return list(self.active_migrations.keys())

    def get_migration_history(self) -> List[Dict]:
        """Get migration history"""
        with self.lock:
            return [
                {
                    'migration_id': mid,
                    'container_id': m['request'].container_id,
                    'state': m['state'].value,
                    'start_time': m['start_time'],
                    'completion_time': m.get('completion_time'),
                    'success': m['state'] == MigrationState.COMPLETED
                }
                for mid, m in self.migration_history.items()
            ]

    def cleanup(self):
        """Cleanup resources"""
        self.ncm.stop_monitoring()
        logger.info("Migration Controller cleaned up successfully")