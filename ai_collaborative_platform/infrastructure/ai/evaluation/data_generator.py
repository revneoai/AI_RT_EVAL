from typing import List, Dict, Any
import numpy as np
from datetime import datetime, timedelta

class DataGenerator:
    """Generate synthetic data for testing"""
    def __init__(self, data_type: str = "property"):
        self.data_type = data_type
        self.anomaly_probability = 0.05
        self.drift_probability = 0.1

    def generate_baseline(self, size: int = 1000) -> List[float]:
        """Generate baseline dataset"""
        if self.data_type == "property":
            # Generate property prices between $10K-$1M
            return list(np.random.lognormal(12, 1, size))
        else:
            # Generate transaction amounts between $10-$10K
            return list(np.random.lognormal(5, 1.5, size))

    def generate_stream(self, drift: bool = False) -> float:
        """Generate single data point"""
        if self.data_type == "property":
            base = np.random.lognormal(12, 1)
            if drift:
                base *= 1.5  # Simulate market shift
        else:
            base = np.random.lognormal(5, 1.5)
            if drift:
                base *= 2  # Simulate transaction pattern shift

        # Add occasional anomalies
        if np.random.random() < self.anomaly_probability:
            base *= 5  # Significant deviation

        return float(base)
