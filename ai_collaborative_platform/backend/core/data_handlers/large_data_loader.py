from typing import Generator, Optional, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path
import dask.dataframe as dd
import psutil
import gc
import math

class LargeDataLoader:
    """Handle large datasets with Dask with dynamic chunking"""
    
    def __init__(self, chunk_size: int = 100_000):
        self.chunk_size = chunk_size
        self.columns = None
        self.dtypes = None
        self.data_path = Path("data")
        self.samples_path = self.data_path / "samples"
        self.processed_path = self.data_path / "processed"
        self.total_chunks = 0
        self.processed_chunks = 0
        self.memory_usage_history = []
        self.baseline_data = None
        self.current_window = []
        self.window_size = 1000  # For drift detection
        
        # Create directories if they don't exist
        self.data_path.mkdir(exist_ok=True)
        self.samples_path.mkdir(exist_ok=True)
        self.processed_path.mkdir(exist_ok=True)
    
    def get_memory_status(self) -> Dict[str, float]:
        """Get current memory usage"""
        mem = psutil.virtual_memory()
        return {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "percent_used": mem.percent
        }
    
    def infer_dtypes(self, file_path: str) -> Dict[str, str]:
        """Infer correct dtypes from sample data"""
        # Read a small sample to infer types
        sample = pd.read_csv(file_path, nrows=1000)
        
        dtypes = {}
        for col in sample.columns:
            # Check if column contains Arabic text
            has_arabic = sample[col].astype(str).str.contains(r'[\u0600-\u06FF]').any()
            # Check if column name suggests text content
            is_text_col = any(text in col.lower() for text in ['name', 'type', 'project', 'rooms', '_ar', '_en'])
            
            if has_arabic or is_text_col:
                dtypes[col] = 'object'
            elif sample[col].dtype == 'int64':
                dtypes[col] = 'int64'
            elif sample[col].dtype == 'float64':
                dtypes[col] = 'float64'
            else:
                dtypes[col] = 'object'
        
        return dtypes
    
    def optimize_chunk_size(self, sample_df: pd.DataFrame) -> int:
        """Dynamically optimize chunk size based on system capabilities"""
        try:
            # Get system memory info
            mem = psutil.virtual_memory()
            available_memory = mem.available
            
            # Calculate memory usage per row
            row_memory = sample_df.memory_usage(deep=True).sum() / len(sample_df)
            
            # Target using 20% of available memory for each chunk
            target_memory = available_memory * 0.2
            
            # Calculate optimal chunk size
            optimal_chunk_size = int(target_memory / row_memory)
            
            # Round to nearest 10,000 for readability
            optimal_chunk_size = math.ceil(optimal_chunk_size / 10000) * 10000
            
            # Set bounds for chunk size
            min_chunk_size = 10000
            max_chunk_size = 500000
            optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))
            
            print(f"Optimized chunk size: {optimal_chunk_size:,} rows")
            print(f"Estimated memory per chunk: {(optimal_chunk_size * row_memory / 1024**2):.1f} MB")
            
            return optimal_chunk_size
            
        except Exception as e:
            print(f"Error optimizing chunk size: {e}")
            return self.chunk_size  # Fallback to initial chunk size
    
    def load_large_csv(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Load large CSV file using Dask with optimized chunking"""
        try:
            # Read a sample to optimize chunk size
            sample = pd.read_csv(file_path, nrows=1000)
            self.chunk_size = self.optimize_chunk_size(sample)
            
            # Infer proper dtypes
            self.dtypes = self.infer_dtypes(file_path)
            
            # Create metadata for Dask
            meta = pd.DataFrame({col: pd.Series(dtype=dtype) 
                               for col, dtype in self.dtypes.items()})
            
            # Calculate optimal blocksize in bytes
            blocksize = self.chunk_size * sample.memory_usage(deep=True).sum() / len(sample)
            
            # Read CSV with optimized parameters
            ddf = dd.read_csv(
                file_path,
                dtype=self.dtypes,
                blocksize=blocksize
            )
            
            self.total_chunks = ddf.npartitions
            self.processed_chunks = 0
            
            # Process chunks with memory monitoring
            for partition in range(ddf.npartitions):
                mem_status = self.get_memory_status()
                self.memory_usage_history.append(mem_status["percent_used"])
                
                # Check if we need to adjust chunk size
                if self.should_adjust_chunk_size():
                    self.adjust_chunk_size()
                
                if mem_status["percent_used"] > 90:
                    print(f"High memory usage ({mem_status['percent_used']}%). Cleaning memory...")
                    gc.collect()
                
                try:
                    chunk = ddf.get_partition(partition).compute()
                    processed_chunk = self._process_chunk(chunk)
                    self.processed_chunks += 1
                    
                    print(f"Processed chunk {self.processed_chunks}/{self.total_chunks} "
                          f"({(self.processed_chunks/self.total_chunks*100):.1f}%)")
                    
                    yield processed_chunk
                    
                    del chunk
                    del processed_chunk
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing partition {partition}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error loading large CSV: {str(e)}")
            yield self._generate_synthetic_data()
    
    def should_adjust_chunk_size(self) -> bool:
        """Determine if chunk size needs adjustment based on memory usage trend"""
        if len(self.memory_usage_history) < 3:
            return False
        
        # Check if memory usage is trending up significantly
        recent_usage = self.memory_usage_history[-3:]
        return (recent_usage[-1] - recent_usage[0]) > 10  # 10% increase threshold
    
    def adjust_chunk_size(self):
        """Adjust chunk size based on memory usage"""
        if self.memory_usage_history[-1] > 80:
            # Reduce chunk size by 25% if memory usage is high
            self.chunk_size = int(self.chunk_size * 0.75)
            print(f"Reducing chunk size to {self.chunk_size:,} due to high memory usage")
        elif self.memory_usage_history[-1] < 50:
            # Increase chunk size by 25% if memory usage is low
            self.chunk_size = int(self.chunk_size * 1.25)
            print(f"Increasing chunk size to {self.chunk_size:,} due to low memory usage")
    
    def save_processed_chunks(self, file_path: str, output_prefix: str = "processed"):
        """Save data in smaller chunks"""
        chunk_num = 0
        
        for chunk in self.load_large_csv(file_path):
            # Save each chunk as parquet (more efficient than CSV)
            output_path = self.processed_path / f"{output_prefix}_{chunk_num}.parquet"
            chunk.to_parquet(output_path)
            chunk_num += 1
            
            # Save a sample for quick testing
            if chunk_num == 1:
                sample_path = self.samples_path / f"{output_prefix}_sample.parquet"
                chunk.head(1000).to_parquet(sample_path)
    
    def get_processed_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """Load processed chunks one at a time"""
        for parquet_file in self.processed_path.glob("*.parquet"):
            yield pd.read_parquet(parquet_file)
    
    def _process_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process each chunk with proper encoding and type handling"""
        chunk = chunk.copy()
        
        # Add metadata columns without converting to numeric
        chunk['processed_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        chunk['is_synthetic'] = False
        
        # Ensure proper encoding for text columns
        for col in chunk.columns:
            if self.dtypes.get(col) == 'object':
                chunk[col] = chunk[col].astype('string')
        
        return chunk
    
    def _generate_synthetic_data(self, size: int = 1000) -> pd.DataFrame:
        """Generate synthetic data matching the schema"""
        if self.columns is None:
            # Default columns if no schema detected
            return pd.DataFrame({
                'value': np.random.lognormal(12, 1, size),
                'is_synthetic': True,
                'processed_timestamp': pd.Timestamp.now()
            })
        
        # Generate data matching the detected schema
        synthetic_data = {}
        for col in self.columns:
            if 'price' in col.lower():
                synthetic_data[col] = np.random.lognormal(12, 1, size)
            elif 'date' in col.lower():
                synthetic_data[col] = pd.date_range(start='2024-01-01', periods=size)
            else:
                synthetic_data[col] = [f'Synthetic_{i}' for i in range(size)]
                
        return pd.DataFrame(synthetic_data)
    
    def get_summary(self, file_path: str) -> Dict[str, Any]:
        """Get file and data summary"""
        try:
            # Get file info
            file_size = Path(file_path).stat().st_size / (1024 * 1024)  # MB
            
            # Read sample for column info
            dtypes = self.infer_dtypes(file_path)
            
            return {
                "file_size_mb": f"{file_size:.1f} MB",
                "columns": list(dtypes.keys()),
                "dtypes": dtypes,
                "arabic_columns": [col for col, dtype in dtypes.items() 
                                 if '_ar' in col.lower()],
                "numeric_columns": [col for col, dtype in dtypes.items() 
                                  if dtype in ('int64', 'float64')]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def set_baseline(self, data: pd.DataFrame):
        """Set baseline data for drift detection"""
        self.baseline_data = data
        
    def update_monitoring_window(self, chunk: pd.DataFrame):
        """Update sliding window for drift detection"""
        self.current_window.extend(chunk.to_dict('records'))
        # Keep window size fixed
        if len(self.current_window) > self.window_size:
            self.current_window = self.current_window[-self.window_size:] 