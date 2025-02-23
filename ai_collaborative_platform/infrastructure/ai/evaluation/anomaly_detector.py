from typing import List, Dict, Any
import numpy as np
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

    def check_anomaly(self, value: float) -> Dict[str, Any]:
        """Check if value is anomalous"""
        self.window.append(value)
        if len(self.window) > self.config.window_size:
            self.window.pop(0)

        mean = np.mean(self.window)
        std = np.std(self.window)
        z_score = (value - mean) / (std if std > 0 else 1)

        return {
            "is_anomaly": abs(z_score) > self.config.z_score_threshold,
            "z_score": z_score,
            "threshold": self.config.z_score_threshold,
            "current_value": value,
            "window_mean": mean,
            "window_std": std
        }
