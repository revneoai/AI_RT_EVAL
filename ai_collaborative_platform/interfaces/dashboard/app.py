import streamlit as st
import plotly.graph_objects as go
import numpy as np
from ai_collaborative_platform.backend.core.evaluation.drift_detector import DriftDetector
from ai_collaborative_platform.backend.core.evaluation.anomaly_detector import AnomalyDetector, AnomalyConfig
from ai_collaborative_platform.backend.core.data_handlers.data_loader import DataLoader
from ai_collaborative_platform.backend.core.data_handlers.large_data_loader import LargeDataLoader
from ai_collaborative_platform.backend.core.data_handlers.progressive_loader import ProgressiveLoader
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import os
import gc
from ai_collaborative_platform.interfaces.dashboard.components.data_loader_view import show_data_loading_options
from ai_collaborative_platform.interfaces.dashboard.components.monitoring_view import show_monitoring_configuration, show_system_metrics
from ai_collaborative_platform.interfaces.dashboard.components.processing_view import (
    show_processing_controls, 
    show_monitoring_results,
    show_drift_results,
    show_anomaly_results
)
import json
from ai_collaborative_platform.interfaces.dashboard.components.export_processing_view import show_export_sidebar


def show_user_guide():
    """Show application usage guide"""
    with st.expander("📚 How to Use This Dashboard", expanded=True):
        st.markdown("""
        ### Quick Start Guide
        1. **Choose Loading Method**:
           - Standard: For files < 200MB
           - Progressive: For medium files with progress tracking
           - Large Data (Dask): For very large files with parallel processing
        
        2. **Upload Your Data**:
           - Small files: Use direct upload
           - Large files: Place in 'data/uploads' folder
        
        3. **Processing Options**:
           - Automatic chunk size optimization
           - Real-time memory monitoring
           - Drift and anomaly detection
        """)

def determine_optimal_chunk_size(file_path: str, available_memory: float) -> int:
    """Determine optimal chunk size based on file size and available memory"""
    file_size = os.path.getsize(file_path)
    sample_df = pd.read_csv(file_path, nrows=1000)
    row_size = file_size / len(sample_df)
    
    # Use 20% of available memory as target chunk memory
    target_chunk_memory = available_memory * 0.2
    optimal_chunk_size = int(target_chunk_memory / row_size)
    
    # Ensure chunk size is within reasonable bounds
    return max(1000, min(optimal_chunk_size, 100000))

def analyze_real_estate_data(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze real estate data with flexible column mapping"""
    analysis = {
        'timestamp': pd.Timestamp.now(),
        'metrics': {},
        'trends': {},
        'anomalies': {},
        'recommendations': []
    }
    
    # Dynamic column mapping
    price_cols = [col for col in data.columns if any(
        term in col.lower() for term in ['price', 'worth', 'value', 'amount']
    )]
    location_cols = [col for col in data.columns if any(
        term in col.lower() for term in ['location', 'area', 'region', 'zone']
    )]
    type_cols = [col for col in data.columns if any(
        term in col.lower() for term in ['type', 'category', 'usage']
    )]
    
    # Basic metrics
    for price_col in price_cols:
        if pd.api.types.is_numeric_dtype(data[price_col]):
            analysis['metrics'][f'{price_col}_stats'] = {
                'mean': data[price_col].mean(),
                'median': data[price_col].median(),
                'std': data[price_col].std(),
                'min': data[price_col].min(),
                'max': data[price_col].max()
            }
    
    # Location analysis if available
    if location_cols:
        for loc_col in location_cols:
            analysis['trends'][f'{loc_col}_distribution'] = (
                data[loc_col].value_counts().head(10).to_dict()
            )
    
    # Property type analysis if available
    if type_cols:
        for type_col in type_cols:
            analysis['trends'][f'{type_col}_distribution'] = (
                data[type_col].value_counts().head(10).to_dict()
            )
    
    return analysis

def export_analysis_report(analysis: Dict[str, Any], format: str = 'csv') -> Tuple[str, str]:
    """Export analysis and return content and filename"""
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"real_estate_analysis_{timestamp}.{format}"
    filepath = export_dir / filename
    
    if format == 'csv':
        # Prepare data for export
        export_data = {
            'timestamp': analysis['timestamp'],
            'market_status': analysis['market_trends']['status'],
            'market_details': analysis['market_trends']['details'],
            'anomalies_detected': analysis['price_anomalies']['count'],
            'drift_score': analysis['drift_analysis']['score'],
            'recommendations': '; '.join(analysis.get('recommendations', []))
        }
        
        # Save to file
        df = pd.DataFrame([export_data])
        df.to_csv(filepath, index=False)
        content = df.to_csv(index=False)
        
    elif format == 'json':
        content = json.dumps(analysis, default=str, indent=2)
        with open(filepath, 'w') as f:
            f.write(content)
    
    return content, str(filepath)

def generate_real_estate_insights(drift_result: Dict, anomalies: List, config: Dict, current_data: List) -> Dict:
    """Generate real estate specific insights from monitoring results"""
    insights = {
        'timestamp': pd.Timestamp.now(),
        'market_trends': {
            'status': 'Stable',
            'details': 'Market prices align with historical patterns',
            'action': 'Continue normal operations'
        },
        'price_anomalies': {
            'count': 0,
            'details': '',
            'action': ''
        },
        'drift_analysis': {
            'status': 'Normal',
            'score': drift_result['drift_score'],
            'action': ''
        },
        'metrics': {},
        'trends': {},
        'recommendations': []
    }
    
    # Analyze drift for market trends
    if drift_result['drift_detected']:
        insights['market_trends'].update({
            'status': '⚠️ Shifting',
            'details': 'Detected significant market price movement',
            'action': 'Consider updating valuation models'
        })
        insights['recommendations'].append(
            "Review pricing models due to market shift"
        )
    
    # Analyze anomalies for price insights
    anomaly_count = sum(1 for a in anomalies if a['is_anomaly'])
    if anomaly_count > 0:
        insights['price_anomalies'].update({
            'count': anomaly_count,
            'details': f'Found {anomaly_count} unusual property prices',
            'action': 'Review pricing strategy'
        })
        insights['recommendations'].append(
            f"Investigate {anomaly_count} anomalous transactions"
        )
    
    # Calculate basic statistics
    if isinstance(current_data, pd.Series):
        data_stats = {
            'mean': current_data.mean(),
            'median': current_data.median(),
            'std': current_data.std(),
            'min': current_data.min(),
            'max': current_data.max()
        }
        insights['metrics']['current_window'] = data_stats
    
    return insights

def show_export_options(insights: Dict[str, Any]):
    """Show export options with clear feedback"""
    st.write("### 💾 Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export as CSV"):
            try:
                content, filepath = export_analysis_report(insights, 'csv')
                st.download_button(
                    "📥 Download CSV",
                    content,
                    f"real_estate_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
                st.success(f"✅ Report exported successfully!")
                st.info(f"📁 File saved to: {filepath}")
                
                # Show preview
                with st.expander("Preview Export"):
                    st.write(pd.read_csv(filepath))
                    
            except PermissionError:
                st.error("❌ Error: Unable to save file. Please check folder permissions.")
            except pd.errors.EmptyDataError:
                st.error("❌ Error: No data to export.")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                st.exception(e)  # This will show the full error trace
    
    with col2:
        if st.button("Export as JSON"):
            try:
                content, filepath = export_analysis_report(insights, 'json')
                st.download_button(
                    "📥 Download JSON",
                    content,
                    f"real_estate_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )
                st.success(f"✅ Report exported successfully!")
                st.info(f"📁 File saved to: {filepath}")
                
                # Show preview
                with st.expander("Preview Export"):
                    st.json(json.loads(content))
                    
            except PermissionError:
                st.error("❌ Error: Unable to save file. Please check folder permissions.")
            except json.JSONDecodeError:
                st.error("❌ Error: Invalid JSON data for export.")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
                st.exception(e)  # This will show the full error trace

def show_real_estate_dashboard(data: pd.DataFrame, insights: Dict):
    """Display real estate focused monitoring dashboard"""
    st.write("### 🏠 Real Estate Market Monitor")
    
    # Market Trends
    with st.expander("📈 Market Trends Analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Market Status",
                insights['market_trends']['status']
            )
            st.write(insights['market_trends']['details'])
        with col2:
            st.write("**Recommended Action:**")
            st.info(insights['market_trends']['action'])
    
    # Price Anomalies
    with st.expander("💰 Price Analysis", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Unusual Transactions",
                insights['price_anomalies']['count']
            )
            st.write(insights['price_anomalies']['details'])
        with col2:
            if insights['price_anomalies']['action']:
                st.write("**Recommended Action:**")
                st.warning(insights['price_anomalies']['action'])
    
    # Market Drift
    with st.expander("🎯 Market Stability", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Market Drift Score",
                f"{insights['drift_analysis']['score']:.3f}"
            )
        with col2:
            if insights['drift_analysis']['action']:
                st.write("**Recommended Action:**")
                st.info(insights['drift_analysis']['action'])
    
    # Recommendations
    if insights['recommendations']:
        st.write("### 📋 Key Recommendations")
        for rec in insights['recommendations']:
            st.write(f"• {rec}")
    
    # Replace old export options with new export processing view
    show_export_sidebar(insights)

def process_data(file_path: str, loader: LargeDataLoader, drift_detector: DriftDetector, 
                anomaly_detector: AnomalyDetector, config: Dict[str, Any]):
    """Process data with monitoring"""
    # Create containers for UI elements
    baseline_progress = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    preview_container = st.container()
    monitoring_container = st.container()
    stats_container = st.container()
    
    try:
        # First pass: Establish baseline
        baseline_progress.info("📊 Establishing baseline...")
        baseline_chunks = []
        baseline_bar = st.progress(0)
        
        for i, chunk in enumerate(loader.load_large_csv(file_path)):
            baseline_chunks.append(chunk[config['drift']['monitoring_col']].tolist())
            baseline_bar.progress(min((i + 1) / 5, 1.0))
            
            with stats_container:
                st.write(f"📈 Baseline chunk {i+1}: {len(chunk)} rows processed")
                st.write(f"📊 Column being monitored: {config['drift']['monitoring_col']}")
                
            if len(baseline_chunks) >= 5:
                break
        
        baseline_progress.success("✅ Baseline established!")
        st.write(f"📊 Total baseline samples: {sum(len(chunk) for chunk in baseline_chunks)}")
        baseline_bar.empty()
        
        # Set baseline for drift detection
        baseline_data = [val for chunk in baseline_chunks for val in chunk]
        drift_detector.set_baseline(baseline_data)
        
        # Second pass: Monitor drift and anomalies
        st.write("### 🔍 Real-time Monitoring")
        
        # Create metrics containers
        col1, col2, col3 = st.columns(3)
        total_rows = 0
        total_anomalies = 0
        max_drift = 0
        
        for chunk_num, chunk in enumerate(loader.load_large_csv(file_path)):
            if st.session_state.get('stop_processing', False):
                st.warning("⏹️ Processing stopped by user")
                break
            
            # Update progress
            progress = min((chunk_num + 1) / loader.total_chunks, 1.0)
            progress_bar.progress(progress)
            status_text.write(f"⏳ Processing chunk {chunk_num + 1}/{loader.total_chunks}")
            
            # Process current chunk
            current_data = chunk[config['drift']['monitoring_col']].tolist()
            total_rows += len(current_data)
            
            # Check for drift
            drift_result = drift_detector.check_drift(current_data)
            max_drift = max(max_drift, drift_result['drift_score'])
            
            # Check for anomalies
            anomalies = [
                anomaly_detector.check_anomaly(value) 
                for value in current_data
            ]
            chunk_anomalies = sum(1 for a in anomalies if a['is_anomaly'])
            total_anomalies += chunk_anomalies
            
            # Show monitoring results
            with monitoring_container:
                # Generate and show real estate insights
                insights = generate_real_estate_insights(
                    drift_result,
                    anomalies,
                    config,
                    pd.Series(current_data)
                )
                show_real_estate_dashboard(pd.DataFrame(current_data), insights)
            
            # Clean up
            del chunk
            gc.collect()
        
        # Show final summary
        st.success("✅ Processing complete!")
        st.write("### 📊 Final Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("Total Rows Processed", f"{total_rows:,}")
        with summary_col2:
            st.metric("Total Anomalies Found", f"{total_anomalies:,}")
        with summary_col3:
            st.metric("Max Drift Score", f"{max_drift:.3f}")
            
        # Store insights in session state for export
        st.session_state.current_insights = insights
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.exception(e)  # This will show the full traceback

def show_dask_view(loader: LargeDataLoader, file_path: str):
    """Dask-based view for very large files with monitoring"""
    st.write("### Processing with Dask")
    
    # Get actual columns from the data
    sample_df = pd.read_csv(file_path, nrows=5)
    numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # System Status and Configuration
    with st.expander("System & Monitoring Configuration", expanded=True):
        config = show_monitoring_configuration(numeric_cols, loader)
    
    # Memory Monitoring
    memory_trend = st.empty()
    
    # Processing controls
    start, stop, clear = show_processing_controls()
    
    if clear:
        gc.collect()
        loader.current_window = []
        loader.memory_usage_history = []
        st.success("Memory cleared")
    
    if start:
        progress_bar = st.progress(0)
        status_text = st.empty()
        preview_container = st.container()
        monitoring_container = st.container()
        memory_container = st.container()
        
        try:
            # First pass: Establish baseline
            st.info("📊 Establishing baseline...")
            baseline_chunks = []
            for chunk in loader.load_large_csv(file_path):
                baseline_chunks.append(chunk)
                if len(baseline_chunks) >= 5:  # Use first 5 chunks for baseline
                    break
                
                # Update memory history
                mem_status = loader.get_memory_status()
                loader.memory_usage_history.append(mem_status['percent_used'])
            
            baseline_data = pd.concat(baseline_chunks)
            
            # Initialize detectors
            drift_detector = DriftDetector(threshold=config['drift']['threshold'])
            anomaly_detector = AnomalyDetector(
                config=AnomalyConfig(
                    z_score_threshold=config['anomaly']['zscore'],
                    window_size=config['drift']['window_size']
                )
            )
            
            drift_detector.set_baseline(baseline_data[config['drift']['monitoring_col']].tolist())
            
            # Second pass: Monitor drift and anomalies
            for chunk_num, chunk in enumerate(loader.load_large_csv(file_path)):
                if stop:
                    st.warning("⏹️ Processing stopped by user")
                    break
                
                # Update progress and memory metrics
                progress = min((chunk_num + 1) / loader.total_chunks, 1.0)
                progress_bar.progress(progress)
                
                mem_status = loader.get_memory_status()
                loader.memory_usage_history.append(mem_status['percent_used'])
                
                # Update status with memory info
                status_text.text(
                    f"📈 Processing chunk {chunk_num + 1}/{loader.total_chunks} "
                    f"(Memory: {mem_status['percent_used']}%, "
                    f"Chunk: {loader.chunk_size:,} rows)"
                )
                
                # Show memory metrics
                with memory_container:
                    col1, col2, col3 = st.columns(3)
                    col1.metric(
                        "Memory Used %", 
                        f"{mem_status['percent_used']}%",
                        delta=f"{mem_status['percent_used'] - loader.memory_usage_history[-2]:.1f}" 
                        if len(loader.memory_usage_history) > 1 else None
                    )
                    col2.metric("Chunk Size", f"{loader.chunk_size:,}")
                    col3.metric("Chunks Processed", f"{chunk_num + 1}/{loader.total_chunks}")
                
                # Update memory trend plot
                with memory_trend:
                    st.write("### 📊 Memory Usage Trend")
                    if len(loader.memory_usage_history) > 1:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=loader.memory_usage_history,
                            name="Memory Usage %"
                        ))
                        st.plotly_chart(fig)
                
                # Process current chunk
                current_data = chunk[config['drift']['monitoring_col']].tolist()
                
                # Check for drift
                drift_result = drift_detector.check_drift(current_data)
                
                # Check for anomalies
                anomalies = [
                    anomaly_detector.check_anomaly(value) 
                    for value in current_data
                ]
                
                # Show monitoring results
                with monitoring_container:
                    st.write("### 🔍 Real-time Monitoring")
                    show_monitoring_results(
                        drift_result=drift_result,
                        anomalies=anomalies,
                        threshold=config['drift']['threshold']
                    )
                
                # Clean up
                del chunk
                gc.collect()
                
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")

def show_data_loading_options():
    """Show data loading interface"""
    with st.sidebar:
        st.write("### 📂 Data Loading Options")
        
        # Loading method selection
        load_method = st.radio(
            "Select Loading Method",
            ["Standard", "Progressive", "Large Data (Dask)"],
            help="""
            - Standard: For files < 200MB
            - Progressive: For medium files with progress tracking
            - Large Data: For very large files with parallel processing
            """
        )
        
        st.markdown("---")
        
        # File selection
        st.write("### 📁 File Selection")
        use_direct_path = st.checkbox(
            "Load from local path",
            help="Use for files >200MB"
        )
        
        file_path = None
        if use_direct_path:
            # Local path option
            upload_dir = ensure_upload_dir()
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
            else:
                st.warning("⚠️ No CSV files found in data/uploads directory")
            
            st.info(f"📁 Upload Directory:\n{upload_dir}")
        else:
            # Standard upload option
            st.info("📤 Standard upload (limit: 200MB)")
            uploaded_file = st.file_uploader(
                "Upload CSV file", 
                type="csv"
            )
            
            if uploaded_file:
                upload_dir = ensure_upload_dir()
                file_path = str(upload_dir / "temp_upload.csv")
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
                st.write(f"📊 File size: {file_size_mb:.1f} MB")
        
        # Advanced options
        if file_path:
            st.markdown("---")
            st.write("### ⚙️ Advanced Options")
            chunk_size = st.number_input(
                "Chunk Size (rows)",
                min_value=1000,
                max_value=100000,
                value=10000,
                step=1000,
                help="Number of rows to process at once"
            )
        else:
            chunk_size = 10000
        
        return {
            'file_path': file_path,
            'method': load_method,
            'chunk_size': chunk_size
        }

def show_monitoring_configuration(numeric_cols, loader) -> Dict[str, Any]:
    """Show monitoring configuration interface"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("💻 System Resources")
        mem_status = loader.get_memory_status()
        st.metric("Total Memory (GB)", f"{mem_status['total_gb']:.1f}")
        st.metric("Available Memory (GB)", f"{mem_status['available_gb']:.1f}")
        st.metric("Memory Used %", f"{mem_status['percent_used']}%")
        st.metric("Optimal Chunk Size", f"{loader.chunk_size:,} rows")
    
    with col2:
        st.info("🎯 Drift Detection")
        drift_threshold = st.slider(
            "Drift Threshold", 
            0.01, 1.0, 0.1,
            help="KL divergence threshold for drift detection"
        )
        monitoring_col = st.selectbox(
            "Monitor Column",
            numeric_cols,
            help="Select column to monitor"
        )
        window_size = st.number_input(
            "Window Size",
            min_value=100,
            max_value=10000,
            value=1000,
            help="Number of records for drift calculation"
        )
    
    with col3:
        st.info("⚠️ Anomaly Detection")
        anomaly_zscore = st.slider(
            "Anomaly Z-Score", 
            1.0, 5.0, 3.0,
            help="Z-score threshold for anomalies"
        )
        alert_threshold = st.number_input(
            "Alert Threshold",
            min_value=1,
            max_value=100,
            value=10,
            help="Minimum anomalies to trigger alert"
        )
    
    return {
        'drift': {
            'threshold': drift_threshold,
            'monitoring_col': monitoring_col,
            'window_size': window_size
        },
        'anomaly': {
            'zscore': anomaly_zscore,
            'alert_threshold': alert_threshold
        },
        'memory': mem_status
    }

def create_dashboard():
    st.title("AI Evaluation Dashboard")
    
    # Initialize session state
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False
    
    # Show user guide
    show_user_guide()
    
    # Data Loading from sidebar
    loading_config = show_data_loading_options()
    
    if loading_config['file_path']:
        try:
            # Get memory status for chunk size optimization
            loader = LargeDataLoader()
            mem_status = loader.get_memory_status()
            
            # Determine optimal chunk size
            optimal_chunk_size = determine_optimal_chunk_size(
                loading_config['file_path'],
                mem_status['available_gb']
            )
            
            # Update loader with optimal chunk size
            loader.chunk_size = optimal_chunk_size
            
            # Get data columns
            sample_df = pd.read_csv(loading_config['file_path'], nrows=5)
            numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_cols:
                st.warning("⚠️ No numeric columns found in the data")
                return
            
            # Configuration
            with st.expander("System & Monitoring Configuration", expanded=True):
                config = show_monitoring_configuration(numeric_cols, loader)
                
                if not config or 'drift' not in config:
                    st.error("❌ Error: Invalid configuration")
                    return
            
            # Initialize detectors
            drift_detector = DriftDetector(threshold=config['drift']['threshold'])
            anomaly_detector = AnomalyDetector(
                config=AnomalyConfig(
                    z_score_threshold=config['anomaly']['zscore'],
                    window_size=config['drift']['window_size']
                )
            )
            
            # Processing controls
            start, stop, clear = show_processing_controls()
            
            if start:
                st.session_state.stop_processing = False
                process_data(
                    file_path=loading_config['file_path'],
                    loader=loader,
                    drift_detector=drift_detector,
                    anomaly_detector=anomaly_detector,
                    config=config
                )
            elif stop:
                st.session_state.stop_processing = True
            elif clear:
                gc.collect()
                st.success("🧹 Memory cleared")
                
            # Add export options to sidebar after processing
            if 'current_insights' in st.session_state:
                show_export_sidebar(st.session_state.current_insights)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def show_progressive_view(loader: ProgressiveLoader, file_path: str):
    """Progressive loading view with detailed feedback"""
    col1, col2 = st.columns(2)
    
    with col1:
        summary = loader.get_summary(file_path)
        st.write("### File Summary")
        for key, value in summary.items():
            st.write(f"- {key}: {value}")
    
    if st.button("Start Loading"):
        # Container for live updates
        preview_container = st.container()
        stats_container = st.container()
        
        with st.spinner("Processing data..."):
            for chunk_num, chunk in enumerate(loader.load_with_progress(file_path)):
                # Update preview
                with preview_container:
                    st.write(f"### Preview (Chunk {chunk_num + 1})")
                    st.dataframe(chunk.head())
                
                # Update stats
                with stats_container:
                    st.write("### Current Statistics")
                    st.write(f"Rows processed: {loader.loaded_rows:,}")
                    st.write(f"Memory usage: {loader.get_memory_usage()['rss_mb']:.1f} MB")

def show_standard_view(loader: DataLoader, file_path: str):
    """Standard view for smaller files"""
    if st.button("Load Data"):
        with st.spinner("Loading data..."):
            for df in loader.load_file(file_path):
                st.write("### Data Preview")
                st.dataframe(df.head())
                
                st.write("### Data Summary")
                summary = loader.get_summary(file_path)
                for key, value in summary.items():
                    st.write(f"- {key}: {value}")
                
                if loader.synthetic_ratio > 0:
                    st.info(f"Using {loader.synthetic_ratio*100}% synthetic data")

def ensure_upload_dir():
    """Ensure upload directory exists"""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir

# Call this when starting the app
ensure_upload_dir()

if __name__ == "__main__":
    create_dashboard() 