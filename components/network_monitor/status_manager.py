
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import threading
from enum import Enum


class NetworkState(Enum):
    """Network state enumeration"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

@dataclass
class NetworkStatus:
    """Network status data structure"""
    state: NetworkState
    metrics: Dict[str, float]
    alerts: List[Dict[str, Any]]
    actions: List[Dict[str, Any]]
    timestamp: datetime

class StatusManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Status Manager
        
        Args:
            config: Configuration dictionary containing:
                - update_interval: Status update interval
                - alert_thresholds: Threshold configurations
                - action_policies: Mitigation action policies
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize manager components"""
        self.current_state = NetworkState.UNKNOWN
        self.status_history = []
        self.alert_history = []
        self.action_history = []
        
        # Initialize update thread
        self.update_interval = self.config.get('update_interval', 60)
        self.is_running = threading.Event()
        self.update_thread = None
        
        # Load configurations
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        self.action_policies = self.config.get('action_policies', {})

    def start_monitoring(self):
        """Start status monitoring"""
        try:
            if not self.is_running.is_set():
                self.is_running.set()
                self.update_thread = threading.Thread(
                    target=self._monitoring_loop
                )
                self.update_thread.start()
                self.logger.info("Status monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            raise

    def stop_monitoring(self):
        """Stop status monitoring"""
        try:
            if self.is_running.is_set():
                self.is_running.clear()
                if self.update_thread:
                    self.update_thread.join()
                self.logger.info("Status monitoring stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {str(e)}")
            raise

    def update_status(self, 
                     analysis_result: PerformanceAnalysis) -> NetworkStatus:
        """
        Update network status based on analysis
        
        Args:
            analysis_result: Results from performance analysis
            
        Returns:
            Updated NetworkStatus object
        """
        try:
            # Determine new state
            new_state = self._determine_state(analysis_result)
            
            # Generate alerts
            alerts = self._generate_alerts(analysis_result, new_state)
            
            # Determine actions
            actions = self._determine_actions(new_state, alerts)
            
            # Create status object
            status = NetworkStatus(
                state=new_state,
                metrics=analysis_result.metrics,
                alerts=alerts,
                actions=actions,
                timestamp=datetime.now()
            )
            
            # Update history
            self._update_history(status)
            
            # Trigger actions if needed
            if actions:
                self._execute_actions(actions)
                
            return status
            
        except Exception as e:
            self.logger.error(f"Status update failed: {str(e)}")
            raise

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running.is_set():
            try:
                # Get latest analysis result
                analysis_result = self._get_latest_analysis()
                
                if analysis_result:
                    # Update status
                    status = self.update_status(analysis_result)
                    
                    # Log status
                    self._log_status(status)
                    
                # Wait for next update
                self.is_running.wait(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                continue

    def _determine_state(self, 
                        analysis_result: PerformanceAnalysis) -> NetworkState:
        """Determine network state"""
        try:
            # Check for critical conditions
            if self._check_critical_conditions(analysis_result):
                return NetworkState.CRITICAL
                
            # Map analysis status to NetworkState
            status_mapping = {
                "HEALTHY": NetworkState.HEALTHY,
                "WARNING": NetworkState.WARNING,
                "DEGRADED": NetworkState.DEGRADED
            }
            
            return status_mapping.get(
                analysis_result.status,
                NetworkState.UNKNOWN
            )
            
        except Exception as e:
            self.logger.error(f"State determination failed: {str(e)}")
            return NetworkState.UNKNOWN

    def _generate_alerts(self,
                        analysis_result: PerformanceAnalysis,
                        state: NetworkState) -> List[Dict[str, Any]]:
        """Generate alerts based on analysis"""
        alerts = []
        try:
            # Process anomalies
            for anomaly in analysis_result.anomalies:
                if self._should_generate_alert(anomaly):
                    alert = self._create_alert(
                        anomaly,
                        state,
                        "ANOMALY"
                    )
                    alerts.append(alert)
                    
            # Check threshold violations
            for metric, value in analysis_result.metrics.items():
                if self._check_threshold_violation(metric, value):
                    alert = self._create_alert(
                        {'metric': metric, 'value': value},
                        state,
                        "THRESHOLD"
                    )
                    alerts.append(alert)
                    
            return alerts
            
        except Exception as e:
            self.logger.error(f"Alert generation failed: {str(e)}")
            return []

    def _determine_actions(self,
                         state: NetworkState,
                         alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Determine mitigation actions"""
        actions = []
        try:
            # Get policy for current state
            policy = self.action_policies.get(state.value, {})
            
            # Generate actions based on policy
            for alert in alerts:
                action = self._get_mitigation_action(alert, policy)
                if action:
                    actions.append(action)
                    
            return actions
            
        except Exception as e:
            self.logger.error(f"Action determination failed: {str(e)}")
            return []

    def _execute_actions(self, actions: List[Dict[str, Any]]):
        """Execute mitigation actions"""
        for action in actions:
            try:
                self._execute_single_action(action)
            except Exception as e:
                self.logger.error(
                    f"Failed to execute action {action['id']}: {str(e)}"
                )
                continue

    def _execute_single_action(self, action: Dict[str, Any]):
        """Execute single mitigation action"""
        try:
            # Log action
            self.logger.info(
                f"Executing action: {json.dumps(action, default=str)}"
            )
            
            # Execute action based on type
            if action['type'] == 'notification':
                self._send_notification(action)
            elif action['type'] == 'mitigation':
                self._apply_mitigation(action)
            elif action['type'] == 'recovery':
                self._initiate_recovery(action)
                
            # Record action
            self.action_history.append({
                'action': action,
                'timestamp': datetime.now(),
                'status': 'completed'
            })
            
        except Exception as e:
            self.logger.error(f"Action execution failed: {str(e)}")
            raise

    def get_current_status(self) -> Optional[NetworkStatus]:
        """Get current network status"""
        try:
            if self.status_history:
                return self.status_history[-1]
            return None
        except Exception as e:
            self.logger.error(f"Failed to get current status: {str(e)}")
            return None

    def get_status_history(self, 
                         limit: int = None) -> List[NetworkStatus]:
        """Get status history"""
        try:
            if limit:
                return self.status_history[-limit:]
            return self.status_history
        except Exception as e:
            self.logger.error(f"Failed to get status history: {str(e)}")
            return []