import streamlit as st
import plotly.graph_objects as go
import pandas as pd

def show_processing_controls():
    """Show processing control buttons"""
    col1, col2, col3 = st.columns(3)
    with col1:
        start = st.button("▶️ Start Processing")
    with col2:
        stop = st.button("⏹️ Stop Processing")
    with col3:
        clear = st.button("🧹 Clear Memory")
    return start, stop, clear

def show_monitoring_results(drift_result, anomalies, threshold):
    """Show monitoring results"""
    col1, col2 = st.columns(2)
    
    with col1:
        show_drift_results(drift_result, threshold)
    
    with col2:
        show_anomaly_results(anomalies)

def show_drift_results(result, threshold):
    """Show drift detection results"""
    st.metric(
        "Drift Score", 
        f"{result['drift_score']:.3f}",
        delta=f"{result['drift_score'] - threshold:.3f}"
    )
    if result["drift_detected"]:
        st.warning(f"⚠️ Drift detected! Score: {result['drift_score']:.3f}")

def show_anomaly_results(anomalies):
    """Show anomaly detection results"""
    anomaly_count = len(anomalies)
    st.metric("Anomalies Found", anomaly_count)
    if anomaly_count > 0:
        st.error(f"🚨 Found {anomaly_count} anomalies!")
        st.dataframe(pd.DataFrame(anomalies).tail(10)) 