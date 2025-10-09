"""Anomaly detection and alerting system."""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.duck import DuckDBManager


@dataclass
class Anomaly:
    """Detected anomaly."""
    metric_name: str
    date: str
    value: float
    expected_range: tuple
    severity: str  # 'low', 'medium', 'high'
    description: str


class AnomalyDetector:
    """
    Anomaly detection for BI metrics.
    Uses statistical methods to detect outliers.
    """
    
    def __init__(self, db: DuckDBManager, threshold_std: float = 2.0):
        """
        Initialize anomaly detector.
        
        Args:
            db: DuckDB manager
            threshold_std: Standard deviation threshold for anomalies
        """
        self.db = db
        self.threshold_std = threshold_std
    
    def detect_revenue_anomalies(self, days: int = 90) -> List[Anomaly]:
        """
        Detect anomalies in daily revenue.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of detected anomalies
        """
        query = f"""
        SELECT 
            order_date,
            SUM(revenue) as daily_revenue
        FROM fact_orders
        WHERE order_date >= CURRENT_DATE - INTERVAL '{days} days'
        GROUP BY order_date
        ORDER BY order_date
        """
        
        data = self.db.query_df(query)
        
        if data.empty or len(data) < 10:
            return []
        
        return self._detect_zscore_anomalies(
            data,
            'daily_revenue',
            'order_date',
            'Revenue'
        )
    
    def detect_order_anomalies(self, days: int = 90) -> List[Anomaly]:
        """
        Detect anomalies in daily order count.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of detected anomalies
        """
        query = f"""
        SELECT 
            order_date,
            COUNT(DISTINCT order_id) as daily_orders
        FROM fact_orders
        WHERE order_date >= CURRENT_DATE - INTERVAL '{days} days'
        GROUP BY order_date
        ORDER BY order_date
        """
        
        data = self.db.query_df(query)
        
        if data.empty or len(data) < 10:
            return []
        
        return self._detect_zscore_anomalies(
            data,
            'daily_orders',
            'order_date',
            'Order Count'
        )
    
    def _detect_zscore_anomalies(self, data: pd.DataFrame,
                                 metric_col: str,
                                 date_col: str,
                                 metric_name: str) -> List[Anomaly]:
        """
        Detect anomalies using z-score method.
        
        Args:
            data: DataFrame with time series data
            metric_col: Column name for metric values
            date_col: Column name for dates
            metric_name: Human-readable metric name
            
        Returns:
            List of anomalies
        """
        anomalies = []
        
        # Compute z-scores
        data['zscore'] = stats.zscore(data[metric_col])
        
        # Find outliers
        outliers = data[abs(data['zscore']) > self.threshold_std]
        
        # Compute expected range
        mean = data[metric_col].mean()
        std = data[metric_col].std()
        expected_range = (mean - self.threshold_std * std, mean + self.threshold_std * std)
        
        for _, row in outliers.iterrows():
            zscore = abs(row['zscore'])
            
            # Determine severity
            if zscore > 3:
                severity = 'high'
            elif zscore > 2.5:
                severity = 'medium'
            else:
                severity = 'low'
            
            description = f"{metric_name} of {row[metric_col]:.2f} is {zscore:.1f}σ from mean"
            
            anomaly = Anomaly(
                metric_name=metric_name,
                date=str(row[date_col]),
                value=float(row[metric_col]),
                expected_range=expected_range,
                severity=severity,
                description=description
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def get_all_anomalies(self, days: int = 90) -> List[Anomaly]:
        """
        Get all anomalies across metrics.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            List of all detected anomalies
        """
        all_anomalies = []
        
        all_anomalies.extend(self.detect_revenue_anomalies(days))
        all_anomalies.extend(self.detect_order_anomalies(days))
        
        # Sort by severity and date
        severity_order = {'high': 0, 'medium': 1, 'low': 2}
        all_anomalies.sort(key=lambda a: (severity_order[a.severity], a.date))
        
        return all_anomalies


class AlertManager:
    """
    Manage alerts and notifications.
    """
    
    def __init__(self, detector: AnomalyDetector):
        """
        Initialize alert manager.
        
        Args:
            detector: AnomalyDetector instance
        """
        self.detector = detector
        self.alerts: List[Dict[str, Any]] = []
    
    def check_and_alert(self, days: int = 90) -> List[Dict[str, Any]]:
        """
        Check for anomalies and generate alerts.
        
        Args:
            days: Number of days to check
            
        Returns:
            List of alert dictionaries
        """
        anomalies = self.detector.get_all_anomalies(days)
        
        self.alerts = []
        
        for anomaly in anomalies:
            alert = {
                'timestamp': pd.Timestamp.now(),
                'metric': anomaly.metric_name,
                'date': anomaly.date,
                'value': anomaly.value,
                'severity': anomaly.severity,
                'description': anomaly.description,
                'action': self._suggest_action(anomaly)
            }
            
            self.alerts.append(alert)
        
        return self.alerts
    
    def _suggest_action(self, anomaly: Anomaly) -> str:
        """Suggest action based on anomaly."""
        if anomaly.severity == 'high':
            return f"Immediate investigation required for {anomaly.metric_name}"
        elif anomaly.severity == 'medium':
            return f"Review {anomaly.metric_name} trend over next few days"
        else:
            return f"Monitor {anomaly.metric_name} for continued anomalies"
    
    def get_summary(self) -> Dict[str, Any]:
        """Get alert summary."""
        if not self.alerts:
            return {
                'total': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            }
        
        return {
            'total': len(self.alerts),
            'high': sum(1 for a in self.alerts if a['severity'] == 'high'),
            'medium': sum(1 for a in self.alerts if a['severity'] == 'medium'),
            'low': sum(1 for a in self.alerts if a['severity'] == 'low')
        }

