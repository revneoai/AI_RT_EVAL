import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any

def show_monitoring_configuration(numeric_cols, loader) -> Dict[str, Any]:
    """Show monitoring configuration interface"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("💻 System Resources")
        return show_system_metrics(loader)
    
    with col2:
        st.info("🎯 Drift Detection")
        return show_drift_config(numeric_cols)
    
    with col3:
        st.info("⚠️ Anomaly Detection")
        return show_anomaly_config()

def show_system_metrics(loader):
    """Show system resource metrics"""
    try:
        mem_status = loader.get_memory_status()
        st.metric("Total Memory (GB)", f"{mem_status['total_gb']:.1f}")
        st.metric("Available Memory (GB)", f"{mem_status['available_gb']:.1f}")
        st.metric("Memory Used %", f"{mem_status['percent_used']}%")
        return mem_status
    except Exception as e:
        st.error(f"Error getting memory status: {str(e)}")
        return {
            'total_gb': 0,
            'available_gb': 0,
            'percent_used': 0
        }

def show_drift_config(numeric_cols):
    """Show drift detection configuration"""
    config = {
        'threshold': st.slider(
            "Drift Threshold", 
            0.01, 1.0, 0.1,
            help="KL divergence threshold for drift detection"
        ),
        'monitoring_col': st.selectbox(
            "Monitor Column",
            numeric_cols,
            help="Select column to monitor"
        ),
        'window_size': st.number_input(
            "Window Size",
            min_value=100,
            max_value=10000,
            value=1000,
            help="Number of records for drift calculation"
        )
    }
    return config

def show_anomaly_config():
    """Show anomaly detection configuration"""
    config = {
        'zscore': st.slider(
            "Anomaly Z-Score", 
            1.0, 5.0, 3.0,
            help="Z-score threshold for anomalies"
        ),
        'alert_threshold': st.number_input(
            "Alert Threshold",
            min_value=1,
            max_value=100,
            value=10,
            help="Minimum anomalies to trigger alert"
        )
    }
    return config