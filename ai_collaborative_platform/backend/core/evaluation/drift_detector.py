from typing import List, Dict, Any, Union
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

    def compute_kl_divergence(baseline: Union[pd.Series, List], 
                            current: Union[pd.Series, List],
                            bins: int = 50) -> float:
        """
        Compute KL divergence between baseline and current distributions
        
        Args:
            baseline: Baseline data distribution
            current: Current data distribution
            bins: Number of bins for histogram
        
        Returns:
            float: KL divergence score
        """
        try:
            # Convert to numpy arrays if needed
            baseline = np.array(baseline)
            current = np.array(current)
            
            # Calculate histograms
            hist_baseline, bins = np.histogram(baseline, bins=bins, density=True)
            hist_current, _ = np.histogram(current, bins=bins, density=True)
            
            # Add small constant to avoid division by zero
            hist_baseline = hist_baseline + 1e-10
            hist_current = hist_current + 1e-10
            
            # Normalize
            hist_baseline = hist_baseline / hist_baseline.sum()
            hist_current = hist_current / hist_current.sum()
            
            # Compute KL divergence
            kl_div = entropy(hist_current, hist_baseline)
            
            return float(kl_div)
            
        except Exception as e:
            print(f"Error computing KL divergence: {str(e)}")
            return 0.0 