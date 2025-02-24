from typing import Generator, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import streamlit as st

class ProgressiveLoader:
    """Handle large CSV files with progress reporting"""
    
    def __init__(self, chunk_size: int = 100_000):
        self.chunk_size = chunk_size
        self.total_rows = 0
        self.loaded_rows = 0
    
    def count_rows(self, file_path: str) -> int:
        """Count total rows without loading file"""
        with open(file_path, 'r') as f:
            self.total_rows = sum(1 for _ in f) - 1  # Subtract header
        return self.total_rows
    
    def load_with_progress(self, file_path: str) -> Generator[pd.DataFrame, None, None]:
        """Load CSV in chunks with progress bar"""
        # Count total rows first
        total_rows = self.count_rows(file_path)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Use chunked reading
            for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                self.loaded_rows += len(chunk)
                
                # Update progress
                progress = min(self.loaded_rows / total_rows, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Loaded {self.loaded_rows:,} of {total_rows:,} rows ({progress:.1%})")
                
                # Basic memory optimization
                chunk = chunk.copy()  # Prevent DataFrame fragmentation
                
                yield chunk
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise
        
        finally:
            # Clean up
            progress_bar.empty()
            status_text.empty()
    
    def get_summary(self, file_path: str) -> dict:
        """Get file summary without loading entire file"""
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        
        # Read just the first chunk for column info
        first_chunk = next(pd.read_csv(file_path, chunksize=1000))
        
        return {
            "file_size_mb": f"{file_size_mb:.1f} MB",
            "total_rows": f"{self.total_rows:,}",
            "columns": list(first_chunk.columns),
            "chunk_size": f"{self.chunk_size:,} rows"
        } 