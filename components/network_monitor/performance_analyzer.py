
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA

@dataclass
class PerformanceAnalysis:
    """Performance analysis results"""
    status: str
    metrics: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    predictions: Dict[str, float]
    timestamp: datetime

class PerformanceAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Performance Analyzer
        
        Args:
            config: Configuration dictionary containing:
                - analysis_window: Window size for analysis
                - prediction_horizon: Steps ahead to predict
                - anomaly_threshold: Threshold for anomaly detection
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._initialize_components()

    def _initialize_components(self):
        """Initialize analyzer components"""
        self.analysis_window = self.config.get('analysis_window', 100)
        self.prediction_horizon = self.config.get('prediction_horizon', 10)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.95)
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize analysis models"""
        # Anomaly detection model
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        
        # Time series models
        self.time_series_models = {}

    def analyze_performance(self, 
                          metrics_history: List[NetworkMetrics]) -> PerformanceAnalysis:
        """
        Analyze network performance
        
        Args:
            metrics_history: List of historical metrics
            
        Returns:
            PerformanceAnalysis object containing results
        """
        try:
            # Prepare data
            data = self._prepare_data(metrics_history)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(data)
            
            # Analyze trends
            trends = self._analyze_trends(data)
            
            # Make predictions
            predictions = self._make_predictions(data)
            
            # Determine status
            status = self._determine_status(anomalies, trends)
            
            # Compute performance metrics
            performance_metrics = self._compute_performance_metrics(data)
            
            # Create analysis result
            analysis = PerformanceAnalysis(
                status=status,
                metrics=performance_metrics,
                anomalies=anomalies,
                predictions=predictions,
                timestamp=datetime.now()
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {str(e)}")
            raise

    def _prepare_data(self, 
                     metrics_history: List[NetworkMetrics]) -> Dict[str, np.ndarray]:
        """Prepare data for analysis"""
        try:
            data = {
                'latency': np.array([m.latency for m in metrics_history]),
                'bandwidth': np.array([m.bandwidth for m in metrics_history]),
                'packet_loss': np.array([m.packet_loss for m in metrics_history]),
                'jitter': np.array([m.jitter for m in metrics_history]),
                'throughput': np.array([m.throughput for m in metrics_history])
            }
            return data
        except Exception as e:
            self.logger.error(f"Data preparation failed: {str(e)}")
            raise

    def _detect_anomalies(self, data: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        anomalies = []
        try:
            for metric, values in data.items():
                # Reshape for isolation forest
                X = values.reshape(-1, 1)
                
                # Fit and predict
                self.anomaly_detector.fit(X)
                scores = self.anomaly_detector.score_samples(X)
                
                # Detect anomalies
                anomaly_points = np.where(scores < -self.anomaly_threshold)[0]
                
                if len(anomaly_points) > 0:
                    anomalies.append({
                        'metric': metric,
                        'indices': anomaly_points.tolist(),
                        'scores': scores[anomaly_points].tolist(),
                        'values': values[anomaly_points].tolist()
                    })
                    
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            return []

    def _analyze_trends(self, data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Analyze trends in metrics"""
        trends = {}
        try:
            for metric, values in data.items():
                # Compute trend statistics
                trend = {
                    'slope': np.polyfit(range(len(values)), values, 1)[0],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'recent_change': self._compute_recent_change(values)
                }
                trends[metric] = trend
                
            return trends
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {str(e)}")
            return {}

    def _make_predictions(self, data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Make predictions using time series models"""
        predictions = {}
        try:
            for metric, values in data.items():
                if metric not in self.time_series_models:
                    # Initialize ARIMA model
                    model = ARIMA(values, order=(1,1,1))
                    self.time_series_models[metric] = model.fit()
                    
                # Make prediction
                forecast = self.time_series_models[metric].forecast(
                    steps=self.prediction_horizon
                )
                predictions[metric] = forecast.tolist()
                
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            return {}

    def _determine_status(self,
                         anomalies: List[Dict[str, Any]],
                         trends: Dict[str, Dict[str, float]]) -> str:
        """Determine network status"""
        try:
            # Count recent anomalies
            recent_anomalies = sum(len(a['indices']) for a in anomalies)
            
            # Check trends
            negative_trends = sum(
                1 for metric in trends.values()
                if metric['slope'] < 0
            )
            
            # Determine status
            if recent_anomalies > self.config.get('anomaly_limit', 5):
                return "DEGRADED"
            elif negative_trends > len(trends) / 2:
                return "WARNING"
            else:
                return "HEALTHY"
                
        except Exception as e:
            self.logger.error(f"Status determination failed: {str(e)}")
            return "UNKNOWN"

    def _compute_performance_metrics(self, 
                                  data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute aggregated performance metrics"""
        try:
            metrics = {}
            for metric, values in data.items():
                metrics[f"{metric}_mean"] = np.mean(values)
                metrics[f"{metric}_std"] = np.std(values)
                metrics[f"{metric}_trend"] = np.polyfit(
                    range(len(values)), 
                    values, 
                    1
                )[0]
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metric computation failed: {str(e)}")
            return {}

    def _compute_recent_change(self, values: np.ndarray) -> float:
        """Compute recent change in values"""
        try:
            if len(values) < 2:
                return 0.0
                
            recent_window = self.config.get('recent_window', 10)
            recent_values = values[-recent_window:]
            
            if len(recent_values) < 2:
                return 0.0
                
            return (recent_values[-1] - recent_values[0]) / recent_values[0]
            
        except Exception as e:
            self.logger.error(f"Change computation failed: {str(e)}")
            return 0.0