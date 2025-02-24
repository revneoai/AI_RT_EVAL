import streamlit as st
from pathlib import Path
import os
from typing import Optional
import pandas as pd

def ensure_upload_dir() -> Path:
    """Ensure upload directory exists and return path"""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir

def show_data_loading_options() -> Optional[str]:
    """Show data loading interface"""
    st.write("### Data Upload")
    
    # Ensure upload directory exists
    upload_dir = ensure_upload_dir()
    
    use_direct_path = st.checkbox("Load from local path (for files >200MB)")
    
    file_path = None
    if use_direct_path:
        file_path = show_local_path_loader(upload_dir)
    else:
        file_path = show_standard_uploader(upload_dir)
        
    return file_path

def show_local_path_loader(upload_dir: Path) -> Optional[str]:
    """Show interface for loading from local path"""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("📂 Select from data/uploads directory")
        available_files = list(upload_dir.glob("*.csv"))
        
        if available_files:
            selected_file = st.selectbox(
                "Select CSV file",
                options=available_files,
                format_func=lambda x: x.name
            )
            file_path = str(selected_file)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            st.write(f"📊 File size: {file_size_mb:.1f} MB")
            return file_path
        else:
            st.warning("⚠️ No CSV files found in data/uploads directory")
            return None
    
    with col2:
        st.info("📁 Upload Directory")
        st.code(str(upload_dir))
        return None

def show_standard_uploader(upload_dir: Path) -> Optional[str]:
    """Show interface for standard file upload"""
    st.info("📤 Standard upload (limit: 200MB)")
    uploaded_file = st.file_uploader(
        "Upload CSV file", 
        type="csv"
    )
    
    if uploaded_file:
        # Create temp directory if it doesn't exist
        temp_dir = upload_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = str(temp_dir / "temp_upload.csv")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.write(f"📊 File size: {file_size_mb:.1f} MB")
        return file_path
    return None

def get_file_info(file_path: str) -> dict:
    """Get file information"""
    if not file_path:
        return {}
        
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
        
    return {
        'name': file_path.name,
        'size_mb': file_path.stat().st_size / (1024 * 1024),
        'modified': file_path.stat().st_mtime,
        'directory': str(file_path.parent)
    } 