from typing import List, Dict, Any
import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div
import pandas as pd

class DriftDetector:
    """Detect distribution drift in real-time data"""
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.baseline_distribution = None
        self.drift_scores = []  # Store historical drift scores

    def set_baseline(self, data: List[float]):
        """Set baseline distribution"""
        self.baseline_distribution = self._compute_distribution(data)

    def check_drift(self, current_data: List[float]) -> Dict[str, Any]:
        """Check for drift in current data"""
        if self.baseline_distribution is None:
            raise ValueError("Baseline distribution not set")
            
        current_distribution = self._compute_distribution(current_data)
        drift_score = self._compute_kl_divergence(
            self.baseline_distribution, 
            current_distribution
        )
        
        self.drift_scores.append(drift_score)  # Store for visualization
        
        return {
            "drift_detected": drift_score > self.threshold,
            "drift_score": drift_score,
            "threshold": self.threshold
        }

    def get_drift_scores(self) -> List[float]:
        """Get historical drift scores for visualization"""
        return self.drift_scores

    def _compute_distribution(self, data: List[float]) -> np.ndarray:
        """Compute histogram distribution of data"""
        hist, _ = np.histogram(data, bins=50, density=True)
        return hist + 1e-10  # Add small constant to avoid zero probabilities

    def _compute_kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute KL divergence between two distributions"""
        return float(np.sum(kl_div(p, q))) 