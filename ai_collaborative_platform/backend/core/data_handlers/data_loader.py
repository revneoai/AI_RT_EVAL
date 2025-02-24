from typing import Generator, Dict, Any, Optional, List
import pandas as pd
import numpy as np
from pathlib import Path
from .base_loader import BaseLoader, DataLoadingError

class DataLoader(BaseLoader):
    """Standard loader for small/medium files with synthetic data capabilities"""
    
    def __init__(self, 
                 chunk_size: int = 100_000, 
                 synthetic_ratio: float = 0.2,
                 required_columns: Optional[List[str]] = None,
                 data_type: str = "property"):
        super().__init__(chunk_size)
        self.synthetic_ratio = synthetic_ratio
        self.real_data: Optional[pd.DataFrame] = None
        self.required_columns = required_columns or ['price', 'location', 'bedrooms']
        self.data_stats: Dict[str, Any] = {}
        self.data_type = data_type
    
    def load_file(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Load file with validation and synthetic data mixing"""
        try:
            # Check file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Monitor memory before loading
            initial_memory = self.get_memory_usage()
            
            # Load and validate data
            df = pd.read_csv(file_path)
            if not self.validate_columns(df, self.required_columns):
                self.logger.warning(f"Missing required columns: {self.required_columns}")
                df = self._add_missing_columns(df)
            
            # Store data stats
            self.data_stats = self.get_data_quality_metrics(df)
            self.real_data = df
            
            # Mix with synthetic data if needed
            if self.synthetic_ratio > 0:
                df = self._mix_with_synthetic(df)
            
            # Monitor memory after loading
            final_memory = self.get_memory_usage()
            memory_increase = final_memory["rss_mb"] - initial_memory["rss_mb"]
            self.logger.info(f"Memory usage increased by {memory_increase:.1f} MB")
            
            yield df
            
        except Exception as e:
            self.logger.error(f"Error loading file: {e}")
            self.logger.info("Falling back to synthetic data")
            yield self.generate_synthetic_data(size=self.chunk_size)
    
    def get_summary(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive data summary"""
        info = self.get_file_info(file_path)
        
        try:
            df = next(self.load_file(file_path))
            info.update({
                "row_count": len(df),
                "column_count": len(df.columns),
                "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                "data_quality": self.data_stats,
                "synthetic_ratio": f"{self.synthetic_ratio * 100}%",
                "columns": list(df.columns)
            })
        except Exception as e:
            self.logger.error(f"Error getting summary: {e}")
            info.update({"error": str(e)})
        
        return info
    
    def _mix_with_synthetic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mix real data with synthetic data"""
        synthetic_size = int(len(df) * self.synthetic_ratio)
        synthetic_df = self.generate_synthetic_data(size=synthetic_size)
        
        # Ensure columns match
        synthetic_df = synthetic_df[df.columns]
        
        # Combine and shuffle
        mixed_df = pd.concat([df, synthetic_df], ignore_index=True)
        return mixed_df.sample(frac=1).reset_index(drop=True)
    
    def _add_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add missing required columns with synthetic data"""
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        
        for col in missing_cols:
            if col == 'price':
                df[col] = np.random.lognormal(12, 1, len(df))
            elif col == 'bedrooms':
                df[col] = np.random.randint(1, 6, len(df))
            else:
                df[col] = 'Unknown'
        
        return df
    
    def get_data_preview(self, df: pd.DataFrame, rows: int = 5) -> Dict[str, Any]:
        """Get data preview with basic statistics"""
        return {
            "head": df.head(rows).to_dict(),
            "dtypes": df.dtypes.to_dict(),
            "description": df.describe().to_dict(),
            "missing_percentages": (df.isnull().sum() / len(df) * 100).to_dict()
        }

    def get_mixed_data(self, size: int = 1000) -> pd.DataFrame:
        """Get a mix of real and synthetic data"""
        if self.real_data is not None:
            # Use real data if available
            real_sample = self.real_data.sample(
                n=min(size, len(self.real_data)), 
                replace=True
            )
            
            # Calculate how many synthetic samples needed
            synthetic_size = size - len(real_sample)
            if synthetic_size > 0:
                synthetic_data = self.generate_synthetic_data(size=synthetic_size)
                # Ensure columns match
                synthetic_data = synthetic_data[real_sample.columns]
                # Combine data
                mixed_data = pd.concat([real_sample, synthetic_data], ignore_index=True)
                return mixed_data.sample(frac=1)  # Shuffle
            return real_sample
        else:
            # If no real data, return synthetic
            return self.generate_synthetic_data(size=size) 