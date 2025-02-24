from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class AnomalyConfig:
    z_score_threshold: float = 3.0
    window_size: int = 100

class AnomalyDetector:
    """Detect anomalies using Z-score method"""
    def __init__(self, config: AnomalyConfig = None):
        self.config = config or AnomalyConfig()
        self.window: List[float] = []
        self.anomalies: List[Dict[str, Any]] = []  # Store anomalies for visualization

    def check_anomaly(self, value: float) -> Dict[str, Any]:
        """Check if value is anomalous"""
        self.window.append(value)
        if len(self.window) > self.config.window_size:
            self.window.pop(0)

        mean = np.mean(self.window)
        std = np.std(self.window)
        z_score = (value - mean) / (std if std > 0 else 1)

        result = {
            "is_anomaly": abs(z_score) > self.config.z_score_threshold,
            "z_score": z_score,
            "threshold": self.config.z_score_threshold,
            "current_value": value,
            "window_mean": mean,
            "window_std": std
        }

        if result["is_anomaly"]:
            self.anomalies.append(result)

        return result

    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get detected anomalies for visualization"""
        return self.anomalies

    def detect_anomalies(data: Union[pd.Series, List], 
                        zscore_threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect anomalies using Z-score method
        
        Args:
            data: Input data series
            zscore_threshold: Z-score threshold for anomaly detection
        
        Returns:
            pd.DataFrame: Detected anomalies with their values and z-scores
        """
        try:
            # Convert to numpy array if needed
            values = np.array(data)
            
            # Calculate z-scores
            mean = np.mean(values)
            std = np.std(values)
            zscores = np.abs((values - mean) / std)
            
            # Find anomalies
            anomaly_indices = np.where(zscores > zscore_threshold)[0]
            
            if len(anomaly_indices) > 0:
                anomalies = pd.DataFrame({
                    'value': values[anomaly_indices],
                    'zscore': zscores[anomaly_indices],
                    'index': anomaly_indices
                })
                return anomalies.sort_values('zscore', ascending=False)
            
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error detecting anomalies: {str(e)}")
            return pd.DataFrame() 