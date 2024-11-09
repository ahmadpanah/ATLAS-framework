# components/network_monitor/metrics_collector.py

import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime
import asyncio
import pandas as pd
from scipy import stats
from collections import deque
import psutil
import aiohttp
from prometheus_client import Counter, Gauge, Histogram
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import torch
import warnings
warnings.filterwarnings('ignore')

class MetricsCollector:
    """Advanced network metrics collection and analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.window_size = config.get('window_size', 1000)
        self.metrics_buffer = {}
        self.scaler = StandardScaler()
        self._initialize_metrics()
        self._setup_prometheus_metrics()

    def _initialize_metrics(self):
        """Initialize metrics collection buffers"""
        try:
            self.metrics_buffer = {
                'latency': deque(maxlen=self.window_size),
                'bandwidth': deque(maxlen=self.window_size),
                'packet_loss': deque(maxlen=self.window_size),
                'jitter': deque(maxlen=self.window_size),
                'throughput': deque(maxlen=self.window_size),
                'connection_count': deque(maxlen=self.window_size),
                'error_rate': deque(maxlen=self.window_size),
                'retransmission_rate': deque(maxlen=self.window_size)
            }
            
            self.anomaly_detection = AnomalyDetector(self.config)
            self.forecasting = MetricsForecaster(self.config)
            
        except Exception as e:
            self.logger.error(f"Metrics initialization failed: {str(e)}")
            raise

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        try:
            self.prom_metrics = {
                'latency': Histogram(
                    'network_latency_ms',
                    'Network latency in milliseconds',
                    buckets=[5, 10, 25, 50, 100, 250, 500, 1000]
                ),
                'bandwidth': Gauge(
                    'network_bandwidth_mbps',
                    'Network bandwidth in Mbps'
                ),
                'packet_loss': Gauge(
                    'packet_loss_ratio',
                    'Packet loss ratio'
                ),
                'errors': Counter(
                    'network_errors_total',
                    'Total network errors'
                )
            }
            
        except Exception as e:
            self.logger.error(f"Prometheus setup failed: {str(e)}")
            raise

    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive network metrics"""
        try:
            # Basic metrics collection
            basic_metrics = await self._collect_basic_metrics()
            
            # Advanced metrics collection
            advanced_metrics = await self._collect_advanced_metrics()
            
            # Statistical analysis
            stats_analysis = self._perform_statistical_analysis(
                basic_metrics,
                advanced_metrics
            )
            
            # Update buffers
            self._update_metrics_buffers(basic_metrics)
            
            # Detect anomalies
            anomalies = await self.anomaly_detection.detect(
                basic_metrics,
                advanced_metrics
            )
            
            # Generate forecasts
            forecasts = await self.forecasting.forecast()
            
            metrics = {
                'basic': basic_metrics,
                'advanced': advanced_metrics,
                'statistics': stats_analysis,
                'anomalies': anomalies,
                'forecasts': forecasts,
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Metrics collection failed: {str(e)}")
            raise

    async def _collect_basic_metrics(self) -> Dict[str, float]:
        """Collect basic network metrics"""
        try:
            network_stats = psutil.net_io_counters()
            
            metrics = {
                'bytes_sent': network_stats.bytes_sent,
                'bytes_recv': network_stats.bytes_recv,
                'packets_sent': network_stats.packets_sent,
                'packets_recv': network_stats.packets_recv,
                'errin': network_stats.errin,
                'errout': network_stats.errout,
                'dropin': network_stats.dropin,
                'dropout': network_stats.dropout
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Basic metrics collection failed: {str(e)}")
            raise

    async def _collect_advanced_metrics(self) -> Dict[str, float]:
        """Collect advanced network metrics"""
        try:
            # Network quality metrics
            quality_metrics = await self._measure_network_quality()
            
            # Connection metrics
            conn_metrics = await self._measure_connections()
            
            # Protocol metrics
            protocol_metrics = await self._measure_protocol_metrics()
            
            return {
                'quality': quality_metrics,
                'connections': conn_metrics,
                'protocols': protocol_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Advanced metrics collection failed: {str(e)}")
            raise

    def _perform_statistical_analysis(self, 
                                   basic_metrics: Dict[str, float],
                                   advanced_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        try:
            analysis = {}
            
            # Basic statistics
            for metric, values in self.metrics_buffer.items():
                if len(values) > 0:
                    analysis[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'skewness': stats.skew(values),
                        'kurtosis': stats.kurtosis(values),
                        'percentiles': {
                            '25': np.percentile(values, 25),
                            '50': np.percentile(values, 50),
                            '75': np.percentile(values, 75),
                            '90': np.percentile(values, 90),
                            '95': np.percentile(values, 95),
                            '99': np.percentile(values, 99)
                        }
                    }
            
            # Time series analysis
            analysis['time_series'] = self._analyze_time_series()
            
            # Correlation analysis
            analysis['correlations'] = self._analyze_correlations()
            
            # Distribution analysis
            analysis['distributions'] = self._analyze_distributions()
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
            raise

    def _analyze_time_series(self) -> Dict[str, Any]:
        """Perform time series analysis"""
        try:
            analysis = {}
            
            for metric, values in self.metrics_buffer.items():
                if len(values) > 1:
                    # Convert to pandas series
                    series = pd.Series(values)
                    
                    # Trend analysis
                    decomposition = sm.tsa.seasonal_decompose(
                        series,
                        period=min(len(series), 30)
                    )
                    
                    analysis[metric] = {
                        'trend': decomposition.trend.dropna().tolist(),
                        'seasonal': decomposition.seasonal.dropna().tolist(),
                        'residual': decomposition.resid.dropna().tolist(),
                        'stationarity': self._check_stationarity(series),
                        'autocorrelation': self._calculate_autocorrelation(series)
                    }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Time series analysis failed: {str(e)}")
            raise

    def _check_stationarity(self, series: pd.Series) -> Dict[str, Any]:
        """Check time series stationarity"""
        try:
            # Augmented Dickey-Fuller test
            adf_test = sm.tsa.stattools.adfuller(series)
            
            return {
                'adf_statistic': adf_test[0],
                'p_value': adf_test[1],
                'critical_values': adf_test[4],
                'is_stationary': adf_test[1] < 0.05
            }
            
        except Exception as e:
            self.logger.error(f"Stationarity check failed: {str(e)}")
            raise

    def _calculate_autocorrelation(self, 
                                 series: pd.Series,
                                 nlags: int = 40) -> Dict[str, Any]:
        """Calculate autocorrelation"""
        try:
            # ACF and PACF
            acf = sm.tsa.stattools.acf(series, nlags=nlags)
            pacf = sm.tsa.stattools.pacf(series, nlags=nlags)
            
            return {
                'acf': acf.tolist(),
                'pacf': pacf.tolist(),
                'significance_level': 1.96/np.sqrt(len(series))
            }
            
        except Exception as e:
            self.logger.error(f"Autocorrelation calculation failed: {str(e)}")
            raise

    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze metric correlations"""
        try:
            # Convert buffers to DataFrame
            df = pd.DataFrame({
                k: list(v) for k, v in self.metrics_buffer.items()
                if len(v) > 0
            })
            
            # Calculate correlations
            correlations = df.corr()
            
            # Calculate partial correlations
            partial_corr = pd.DataFrame(
                np.linalg.pinv(correlations.values),
                index=correlations.index,
                columns=correlations.columns
            )
            
            return {
                'pearson': correlations.to_dict(),
                'spearman': df.corr(method='spearman').to_dict(),
                'partial': partial_corr.to_dict()
            }
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {str(e)}")
            raise

    def _analyze_distributions(self) -> Dict[str, Any]:
        """Analyze metric distributions"""
        try:
            distributions = {}
            
            for metric, values in self.metrics_buffer.items():
                if len(values) > 0:
                    # Fit normal distribution
                    norm_params = stats.norm.fit(values)
                    
                    # Fit other distributions
                    distributions[metric] = {
                        'normal': {
                            'params': norm_params,
                            'kstest': stats.kstest(values, 'norm', norm_params)
                        },
                        'histogram': np.histogram(values, bins='auto'),
                        'kernel_density': stats.gaussian_kde(values)
                    }
            
            return distributions
            
        except Exception as e:
            self.logger.error(f"Distribution analysis failed: {str(e)}")
            raise