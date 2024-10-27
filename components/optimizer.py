import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
from threading import Lock
from scipy.optimize import minimize
from ..utils.data_structures import (
    SecurityLevel,
    NetworkMetrics,
    ContainerMetrics
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityPerformanceOptimizer:
    """
    Implements the Security-Performance Optimizer component from the paper.
    Balances security requirements with performance constraints using multi-objective optimization.
    """
    def __init__(self):
        self.lock = Lock()
        
        # Performance history for containers
        self.performance_history: Dict[str, List[Dict]] = {}
        
        # Security weights for different security levels
        self.security_weights = {
            SecurityLevel.LOW: 1.0,
            SecurityLevel.MEDIUM: 1.25,
            SecurityLevel.HIGH: 1.5,
            SecurityLevel.CRITICAL: 2.0
        }
        
        # Resource constraints
        self.resource_constraints = {
            'cpu_min': 0.1,    # 10% minimum CPU
            'cpu_max': 1.0,    # 100% maximum CPU
            'memory_min': 0.1, # 10% minimum memory
            'memory_max': 1.0, # 100% maximum memory
            'network_min': 0.1,# 10% minimum network
            'network_max': 1.0 # 100% maximum network
        }
        
        # Optimization parameters
        self.optimization_params = {
            'max_iterations': 100,
            'convergence_threshold': 1e-6,
            'population_size': 50
        }

    def optimize_resources(self,
                         container_id: str,
                         security_level: SecurityLevel,
                         network_metrics: NetworkMetrics,
                         container_metrics: ContainerMetrics) -> Dict:
        """
        Optimize resource allocation based on security and performance requirements
        Implements the multi-objective optimization algorithm from the paper
        """
        try:
            with self.lock:
                # Calculate security multiplier
                security_multiplier = self.security_weights[security_level]
                
                # Calculate network quality
                network_quality = self._calculate_network_quality(network_metrics)
                
                # Get current resource usage
                current_resources = self._get_current_resources(container_metrics)
                
                # Define optimization constraints
                constraints = self._define_constraints(
                    security_level,
                    network_quality,
                    current_resources
                )
                
                # Perform multi-objective optimization
                optimal_allocation = self._optimize_allocation(
                    security_multiplier,
                    network_quality,
                    current_resources,
                    constraints
                )
                
                # Update performance history
                self._update_performance_history(
                    container_id,
                    optimal_allocation,
                    security_level,
                    network_metrics
                )
                
                return {
                    'container_id': container_id,
                    'optimal_allocation': optimal_allocation,
                    'security_multiplier': security_multiplier,
                    'network_quality': network_quality,
                    'optimization_score': self._calculate_optimization_score(
                        optimal_allocation,
                        security_multiplier,
                        network_quality
                    )
                }
                
        except Exception as e:
            logger.error(f"Resource optimization failed: {str(e)}")
            raise

    def _optimize_allocation(self,
                           security_multiplier: float,
                           network_quality: float,
                           current_resources: Dict[str, float],
                           constraints: List[Dict]) -> Dict[str, float]:
        """
        Perform multi-objective optimization using SLSQP
        """
        # Initial guess based on current resources
        x0 = [
            current_resources['cpu'],
            current_resources['memory'],
            current_resources['network']
        ]
        
        # Define bounds for each resource
        bounds = [
            (self.resource_constraints['cpu_min'], self.resource_constraints['cpu_max']),
            (self.resource_constraints['memory_min'], self.resource_constraints['memory_max']),
            (self.resource_constraints['network_min'], self.resource_constraints['network_max'])
        ]
        
        # Perform optimization
        result = minimize(
            fun=lambda x: self._objective_function(
                x,
                security_multiplier,
                network_quality
            ),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': self.optimization_params['max_iterations'],
                'ftol': self.optimization_params['convergence_threshold']
            }
        )
        
        if not result.success:
            logger.warning(f"Optimization did not converge: {result.message}")
        
        # Convert result to allocation dictionary
        return {
            'cpu': float(result.x[0]),
            'memory': float(result.x[1]),
            'network': float(result.x[2])
        }

    def _objective_function(self,
                          x: np.ndarray,
                          security_multiplier: float,
                          network_quality: float) -> float:
        """
        Multi-objective function combining security and performance
        Returns negative value because scipy.minimize minimizes the function
        """
        cpu, memory, network = x
        
        # Security objective (higher is better)
        security_score = security_multiplier * np.min([cpu, memory, network])
        
        # Performance objective (lower resource usage is better)
        resource_usage = np.mean([cpu, memory, network])
        
        # Network performance objective
        network_performance = network_quality * network
        
        # Combined objective (negative because we're minimizing)
        return -(security_score + network_performance - 0.5 * resource_usage)

    def _define_constraints(self,
                          security_level: SecurityLevel,
                          network_quality: float,
                          current_resources: Dict[str, float]) -> List[Dict]:
        """Define optimization constraints based on security and network requirements"""
        constraints = []
        
        # Minimum resource constraints based on security level
        min_resource = {
            SecurityLevel.LOW: 0.1,
            SecurityLevel.MEDIUM: 0.2,
            SecurityLevel.HIGH: 0.3,
            SecurityLevel.CRITICAL: 0.4
        }[security_level]
        
        # Add resource constraints
        constraints.extend([
            {
                'type': 'ineq',
                'fun': lambda x: x[0] - min_resource  # CPU constraint
            },
            {
                'type': 'ineq',
                'fun': lambda x: x[1] - min_resource  # Memory constraint
            },
            {
                'type': 'ineq',
                'fun': lambda x: x[2] - min_resource  # Network constraint
            }
        ])
        
        # Network performance constraint
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: network_quality * x[2] - 0.3  # Minimum network performance
        })
        
        # Resource change rate constraints (prevent dramatic changes)
        max_change = 0.3  # Maximum 30% change
        constraints.extend([
            {
                'type': 'ineq',
                'fun': lambda x: x[0] - current_resources['cpu'] * (1 - max_change)
            },
            {
                'type': 'ineq',
                'fun': lambda x: current_resources['cpu'] * (1 + max_change) - x[0]
            }
        ])
        
        return constraints

    def _calculate_network_quality(self, metrics: NetworkMetrics) -> float:
        """Calculate network quality score between 0 and 1"""
        bandwidth_score = min(1.0, metrics.bandwidth / 100)
        latency_score = max(0.0, 1.0 - metrics.latency / 200)
        packet_loss_score = max(0.0, 1.0 - metrics.packet_loss * 50)
        jitter_score = max(0.0, 1.0 - metrics.jitter / 50)
        
        weights = [0.4, 0.3, 0.2, 0.1]
        quality = np.average(
            [bandwidth_score, latency_score, packet_loss_score, jitter_score],
            weights=weights
        )
        
        return float(quality)

    def _get_current_resources(self, metrics: ContainerMetrics) -> Dict[str, float]:
        """Get current resource usage from container metrics"""
        return {
            'cpu': metrics.cpu_usage / 100.0,
            'memory': metrics.memory_usage / 100.0,
            'network': metrics.network_throughput / 100.0
        }

    def _update_performance_history(self,
                                  container_id: str,
                                  allocation: Dict[str, float],
                                  security_level: SecurityLevel,
                                  network_metrics: NetworkMetrics):
        """Update performance history for a container"""
        if container_id not in self.performance_history:
            self.performance_history[container_id] = []
        
        history = self.performance_history[container_id]
        history.append({
            'timestamp': datetime.now(),
            'allocation': allocation,
            'security_level': security_level,
            'network_quality': self._calculate_network_quality(network_metrics)
        })
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_history[container_id] = [
            entry for entry in history
            if entry['timestamp'] > cutoff_time
        ]

    def _calculate_optimization_score(self,
                                   allocation: Dict[str, float],
                                   security_multiplier: float,
                                   network_quality: float) -> float:
        """Calculate overall optimization score"""
        security_score = security_multiplier * min(allocation.values())
        efficiency_score = 1.0 - np.mean(list(allocation.values()))
        network_score = network_quality * allocation['network']
        
        weights = [0.4, 0.3, 0.3]  # Weights for different aspects
        return np.average(
            [security_score, efficiency_score, network_score],
            weights=weights
        )

    def get_optimization_history(self, container_id: str) -> Optional[List[Dict]]:
        """Get optimization history for a container"""
        with self.lock:
            return self.performance_history.get(container_id)

    def analyze_performance_trends(self, container_id: str) -> Dict:
        """Analyze performance trends for a container"""
        with self.lock:
            if container_id not in self.performance_history:
                return {
                    'trends': None,
                    'recommendations': []
                }
            
            history = self.performance_history[container_id]
            if not history:
                return {
                    'trends': None,
                    'recommendations': []
                }
            
            # Calculate trends
            trends = {
                'cpu': self._calculate_trend([h['allocation']['cpu'] for h in history]),
                'memory': self._calculate_trend([h['allocation']['memory'] for h in history]),
                'network': self._calculate_trend([h['allocation']['network'] for h in history])
            }
            
            # Generate recommendations
            recommendations = self._generate_recommendations(trends)
            
            return {
                'trends': trends,
                'recommendations': recommendations
            }

    def _calculate_trend(self, values: List[float]) -> Dict:
        """Calculate trend statistics"""
        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'trend': float(np.polyfit(np.arange(len(values)), values, 1)[0])
        }

    def _generate_recommendations(self, trends: Dict[str, Dict]) -> List[str]:
        """Generate optimization recommendations based on trends"""
        recommendations = []
        
        for resource, trend in trends.items():
            if trend['trend'] > 0.1:
                recommendations.append(
                    f"Consider increasing {resource} allocation capacity"
                )
            elif trend['std'] > 0.2:
                recommendations.append(
                    f"High {resource} usage variability detected"
                )
        
        return recommendations