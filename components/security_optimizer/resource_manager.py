
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from dataclasses import dataclass
import threading
import time

@dataclass
class ResourceAllocation:
    """Resource allocation result"""
    resources: Dict[str, float]
    priority: int
    constraints: Dict[str, Any]
    timestamp: datetime

class ResourceManager:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Resource Manager
        
        Args:
            config: Configuration dictionary containing:
                - resource_limits: Maximum resource limits
                - allocation_strategy: Resource allocation strategy
                - update_interval: Resource update interval
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize manager components"""
        self.resource_limits = self.config.get('resource_limits', {})
        self.current_allocations = {}
        self.allocation_queue = []
        
        # Initialize monitoring thread
        self.update_interval = self.config.get('update_interval', 5)
        self.is_running = threading.Event()
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring"""
        try:
            if not self.is_running.is_set():
                self.is_running.set()
                self.monitor_thread = threading.Thread(
                    target=self._monitoring_loop
                )
                self.monitor_thread.start()
                self.logger.info("Resource monitoring started")
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {str(e)}")
            raise

    def stop_monitoring(self):
        """Stop resource monitoring"""
        try:
            if self.is_running.is_set():
                self.is_running.clear()
                if self.monitor_thread:
                    self.monitor_thread.join()
                self.logger.info("Resource monitoring stopped")
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {str(e)}")
            raise

    def allocate_resources(self,
                          requirements: Dict[str, float],
                          priority: int) -> Optional[ResourceAllocation]:
        """
        Allocate resources based on requirements
        
        Args:
            requirements: Resource requirements
            priority: Request priority
            
        Returns:
            ResourceAllocation object if successful, None otherwise
        """
        try:
            # Check resource availability
            if self._can_allocate(requirements):
                # Create allocation
                allocation = ResourceAllocation(
                    resources=requirements,
                    priority=priority,
                    constraints=self._get_constraints(requirements),
                    timestamp=datetime.now()
                )
                
                # Update allocations
                self._update_allocations(allocation)
                
                return allocation
            else:
                # Add to queue if resources not available
                self.allocation_queue.append({
                    'requirements': requirements,
                    'priority': priority,
                    'timestamp': datetime.now()
                })
                return None
                
        except Exception as e:
            self.logger.error(f"Resource allocation failed: {str(e)}")
            return None

    def release_resources(self, allocation_id: str):
        """Release allocated resources"""
        try:
            if allocation_id in self.current_allocations:
                allocation = self.current_allocations.pop(allocation_id)
                self._process_queue()
                
        except Exception as e:
            self.logger.error(f"Resource release failed: {str(e)}")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_running.is_set():
            try:
                # Check resource utilization
                self._check_utilization()
                
                # Process queued allocations
                self._process_queue()
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                continue

    def _can_allocate(self, requirements: Dict[str, float]) -> bool:
        """Check if resources can be allocated"""
        try:
            # Calculate current utilization
            current_usage = self._calculate_current_usage()
            
            # Check if allocation possible
            for resource, amount in requirements.items():
                if resource in self.resource_limits:
                    available = (self.resource_limits[resource] - 
                               current_usage.get(resource, 0.0))
                    if amount > available:
                        return False
                        
            return True
            
        except Exception as e:
            self.logger.error(f"Allocation check failed: {str(e)}")
            return False

    def _update_allocations(self, allocation: ResourceAllocation):
        """Update current allocations"""
        try:
            allocation_id = f"alloc_{datetime.now().timestamp()}"
            self.current_allocations[allocation_id] = allocation
            
        except Exception as e:
            self.logger.error(f"Allocation update failed: {str(e)}")

    def _process_queue(self):
        """Process queued allocation requests"""
        try:
            # Sort queue by priority
            self.allocation_queue.sort(key=lambda x: (-x['priority'], x['timestamp']))
            
            # Process queue
            remaining_queue = []
            for request in self.allocation_queue:
                if self._can_allocate(request['requirements']):
                    allocation = ResourceAllocation(
                        resources=request['requirements'],
                        priority=request['priority'],
                        constraints=self._get_constraints(request['requirements']),
                        timestamp=datetime.now()
                    )
                    self._update_allocations(allocation)
                else:
                    remaining_queue.append(request)
                    
            self.allocation_queue = remaining_queue
            
        except Exception as e:
            self.logger.error(f"Queue processing failed: {str(e)}")

    def _check_utilization(self):
        """Check resource utilization"""
        try:
            current_usage = self._calculate_current_usage()
            
            # Check for over-utilization
            for resource, usage in current_usage.items():
                if resource in self.resource_limits:
                    if usage > self.resource_limits[resource]:
                        self._handle_over_utilization(resource)
                        
        except Exception as e:
            self.logger.error(f"Utilization check failed: {str(e)}")

    def _calculate_current_usage(self) -> Dict[str, float]:
        """Calculate current resource usage"""
        try:
            usage = {}
            for allocation in self.current_allocations.values():
                for resource, amount in allocation.resources.items():
                    usage[resource] = usage.get(resource, 0.0) + amount
            return usage
            
        except Exception as e:
            self.logger.error(f"Usage calculation failed: {str(e)}")
            return {}

    def _handle_over_utilization(self, resource: str):
        """Handle resource over-utilization"""
        try:
            # Find lowest priority allocations
            allocations = sorted(
                self.current_allocations.items(),
                key=lambda x: x[1].priority
            )
            
            # Release resources until utilization is within limits
            current_usage = self._calculate_current_usage()[resource]
            limit = self.resource_limits[resource]
            
            for alloc_id, allocation in allocations:
                if current_usage <= limit:
                    break
                if resource in allocation.resources:
                    current_usage -= allocation.resources[resource]
                    self.release_resources(alloc_id)
                    
        except Exception as e:
            self.logger.error(f"Over-utilization handling failed: {str(e)}")

    def _get_constraints(self, 
                        requirements: Dict[str, float]) -> Dict[str, Any]:
        """Get constraints for allocation"""
        return {
            'min_resources': requirements,
            'max_resources': self.resource_limits
        }

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        try:
            current_usage = self._calculate_current_usage()
            return {
                'current_usage': current_usage,
                'limits': self.resource_limits,
                'utilization': {
                    resource: usage / self.resource_limits.get(resource, 1.0)
                    for resource, usage in current_usage.items()
                },
                'queue_length': len(self.allocation_queue),
                'active_allocations': len(self.current_allocations)
            }
        except Exception as e:
            self.logger.error(f"Status retrieval failed: {str(e)}")
            return {}