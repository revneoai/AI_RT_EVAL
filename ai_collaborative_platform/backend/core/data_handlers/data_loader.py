from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

class DataLoader:
    """Handle both real and synthetic data loading"""
    
    def __init__(self, data_type: str = "property"):
        self.data_type = data_type
        self.real_data: Optional[pd.DataFrame] = None
        self.synthetic_ratio = 0.2  # 20% synthetic data mixed with real data
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load real data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            required_cols = ['price'] if self.data_type == 'property' else ['amount']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
                
            self.real_data = df
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Falling back to synthetic data...")
            return self._generate_synthetic_data()
    
    def get_mixed_data(self, size: int = 1000) -> pd.DataFrame:
        """Get mixture of real and synthetic data"""
        if self.real_data is None:
            return self._generate_synthetic_data(size)
            
        synthetic_size = int(size * self.synthetic_ratio)
        real_size = size - synthetic_size
        
        # Sample from real data with replacement if needed
        real_sample = self.real_data.sample(n=real_size, replace=True)
        synthetic_sample = self._generate_synthetic_data(synthetic_size)
        
        return pd.concat([real_sample, synthetic_sample]).sample(frac=1)  # Shuffle
    
    def _generate_synthetic_data(self, size: int = 1000) -> pd.DataFrame:
        """Generate synthetic data based on type"""
        if self.data_type == "property":
            # Property prices between $10K-$1M
            values = np.random.lognormal(12, 1, size)
        else:
            # Transaction amounts between $10-$10K
            values = np.random.lognormal(5, 1.5, size)
            
        return pd.DataFrame({
            'value': values,
            'is_synthetic': True
        }) 