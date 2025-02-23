import streamlit as st
import plotly.graph_objects as go
import numpy as np
from ai_collaborative_platform.backend.core.evaluation.drift_detector import DriftDetector
from ai_collaborative_platform.backend.core.evaluation.anomaly_detector import AnomalyDetector
from ai_collaborative_platform.backend.core.data_handlers.data_loader import DataLoader

def create_dashboard():
    st.title("AI Evaluation Dashboard")
    
    # Data source selection
    st.sidebar.header("Data Source")
    data_type = st.sidebar.selectbox(
        "Select Data Type",
        ["property", "transaction"]
    )
    
    use_real_data = st.sidebar.checkbox("Use Real Data")
    
    if use_real_data:
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file", 
            type="csv",
            help="CSV should contain 'price' column for property or 'amount' for transactions"
        )
    
    # Initialize components
    data_loader = DataLoader(data_type=data_type)
    drift_detector = DriftDetector()
    anomaly_detector = AnomalyDetector()
    
    # Load or generate data
    if use_real_data and uploaded_file:
        data = data_loader.load_csv(uploaded_file)
        st.sidebar.success("Using uploaded data with synthetic augmentation")
    else:
        data = data_loader.get_mixed_data(1000)
        st.sidebar.info("Using synthetic data")
    
    # Initialize session state
    if 'real_time_data' not in st.session_state:
        st.session_state.real_time_data = []
        st.session_state.drift_scores = []
        st.session_state.anomalies = []
        
        # Set baseline from initial data
        baseline_data = data['value'].tolist()[:500]
        drift_detector.set_baseline(baseline_data)

    # Add new data point button
    if st.button('Add New Data Point'):
        # Get new point from mixed data
        new_data = data_loader.get_mixed_data(1)
        new_value = float(new_data['value'].iloc[0])
        is_synthetic = bool(new_data['is_synthetic'].iloc[0])
        
        st.session_state.real_time_data.append(new_value)
        
        # Check for drift
        if len(st.session_state.real_time_data) >= 50:
            drift_result = drift_detector.check_drift(st.session_state.real_time_data[-50:])
            st.session_state.drift_scores.append(drift_result['drift_score'])
            
            if drift_result['drift_detected']:
                st.warning(f"Drift detected! Score: {drift_result['drift_score']:.3f}")
        
        # Check for anomalies
        anomaly_result = anomaly_detector.check_anomaly(new_value)
        if anomaly_result['is_anomaly']:
            st.session_state.anomalies.append(new_value)
            st.error(f"Anomaly detected! Value: {new_value:.2f}")
        
        # Show data source
        source_text = "Synthetic" if is_synthetic else "Real"
        st.info(f"Added {source_text} data point: {new_value:.2f}")

    # Real-time Drift Monitoring
    st.header("Distribution Drift")
    if st.session_state.drift_scores:
        drift_fig = go.Figure()
        drift_fig.add_trace(go.Scatter(y=st.session_state.drift_scores, name="Drift Score"))
        drift_fig.add_hline(y=0.1, name="Threshold")
        st.plotly_chart(drift_fig)
    
    # Anomaly Detection
    st.header("Anomaly Detection")
    if st.session_state.real_time_data:
        anomaly_fig = go.Figure()
        anomaly_fig.add_trace(go.Scatter(
            y=st.session_state.real_time_data, 
            name="Values"
        ))
        if st.session_state.anomalies:
            anomaly_fig.add_trace(go.Scatter(
                y=st.session_state.anomalies,
                mode="markers",
                name="Anomalies",
                marker=dict(size=10, color="red")
            ))
        st.plotly_chart(anomaly_fig)

if __name__ == "__main__":
    create_dashboard() 