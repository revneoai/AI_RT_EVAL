from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import psutil
import logging

class BaseLoader(ABC):
    """Abstract base class for all data loaders"""
    
    def __init__(self, chunk_size: int = 100_000):
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Monitor memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": process.memory_percent()
        }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic file information"""
        path = Path(file_path)
        return {
            "size_mb": path.stat().st_size / (1024 * 1024),
            "modified": path.stat().st_mtime,
            "name": path.name,
            "extension": path.suffix
        }
    
    @abstractmethod
    def load_file(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Load file in chunks"""
        pass
    
    @abstractmethod
    def get_summary(self, file_path: str) -> Dict[str, Any]:
        """Get data summary"""
        pass
    
    def validate_columns(self, df: pd.DataFrame, required_cols: list) -> bool:
        """Validate required columns exist"""
        return all(col in df.columns for col in required_cols)
    
    def generate_synthetic_data(self, size: int = 1000, data_type: str = "property") -> pd.DataFrame:
        """Generate synthetic data"""
        if data_type == "property":
            values = np.random.lognormal(12, 1, size)
            columns = ["price", "location", "bedrooms"]
        else:
            values = np.random.lognormal(5, 1.5, size)
            columns = ["amount", "date", "type"]
            
        df = pd.DataFrame({
            "value": values,
            "is_synthetic": True,
            "timestamp": pd.Timestamp.now()
        })
        
        return df
    
    def get_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate data quality metrics"""
        return {
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": df.nunique().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "row_count": len(df)
        }

class DataLoadingError(Exception):
    """Custom exception for data loading errors"""
    pass
