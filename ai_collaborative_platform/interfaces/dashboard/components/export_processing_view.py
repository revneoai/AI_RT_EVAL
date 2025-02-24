import streamlit as st
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_export_directory() -> Path:
    """Setup and verify export directory"""
    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir

def export_analysis_report(analysis: Dict[str, Any], format: str = 'csv') -> Tuple[str, str]:
    """Export analysis and return content and filename"""
    export_dir = setup_export_directory()
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    filename = f"real_estate_analysis_{timestamp}.{format}"
    filepath = export_dir / filename
    
    try:
        if format == 'csv':
            export_data = {
                'timestamp': analysis['timestamp'],
                'market_status': analysis['market_trends']['status'],
                'market_details': analysis['market_trends']['details'],
                'anomalies_detected': analysis['price_anomalies']['count'],
                'drift_score': analysis['drift_analysis']['score'],
                'recommendations': '; '.join(analysis.get('recommendations', []))
            }
            df = pd.DataFrame([export_data])
            df.to_csv(filepath, index=False)
            content = df.to_csv(index=False)
            
        elif format == 'json':
            content = json.dumps(analysis, default=str, indent=2)
            with open(filepath, 'w') as f:
                f.write(content)
        
        logger.info(f"Successfully exported report to {filepath}")
        return content, str(filepath)
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise

def show_export_sidebar(insights: Dict[str, Any]):
    """Show export options in sidebar with processing status and feedback"""
    with st.sidebar:
        st.markdown("---")  # Visual separator
        st.write("### 💾 Export & Downloads")
        
        # Create persistent state for export status
        if 'export_status' not in st.session_state:
            st.session_state.export_status = {'csv': None, 'json': None}
        if 'export_filepath' not in st.session_state:
            st.session_state.export_filepath = {'csv': None, 'json': None}
        
        # Export buttons
        export_format = st.radio(
            "Choose Export Format",
            ["CSV", "JSON"],
            help="Select the format for your export"
        )
        
        if st.button("📤 Export Analysis"):
            format = export_format.lower()
            with st.spinner(f"Exporting as {format.upper()}..."):
                try:
                    content, filepath = export_analysis_report(insights, format)
                    st.session_state.export_status[format] = 'success'
                    st.session_state.export_filepath[format] = filepath
                except Exception as e:
                    st.session_state.export_status[format] = str(e)
        
        # Show status and downloads section
        if any(status == 'success' for status in st.session_state.export_status.values()):
            st.markdown("---")
            st.write("### 📥 Downloads")
            
            for format in ['csv', 'json']:
                if st.session_state.export_status[format] == 'success':
                    filepath = st.session_state.export_filepath[format]
                    
                    # Success message and file location
                    st.success(f"✅ {format.upper()} Ready!")
                    with st.expander(f"📁 File Details ({format.upper()})"):
                        st.info(f"Location: {filepath}")
                        
                        # Preview option
                        if format == 'csv':
                            if st.button(f"👀 Preview CSV", key=f"preview_{format}"):
                                df = pd.read_csv(filepath)
                                st.dataframe(df)
                        else:  # JSON
                            if st.button(f"👀 Preview JSON", key=f"preview_{format}"):
                                with open(filepath) as f:
                                    st.json(json.load(f))
                    
                    # Download button
                    try:
                        with open(filepath, 'r') as f:
                            st.download_button(
                                f"📥 Download {format.upper()}",
                                f.read(),
                                f"real_estate_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.{format}",
                                f"text/{format}",
                                key=f"download_{format}"
                            )
                    except Exception as e:
                        st.error(f"Error preparing download: {str(e)}")
                
                elif st.session_state.export_status[format]:
                    st.error(f"❌ {format.upper()} Export failed: {st.session_state.export_status[format]}")
        
        # Export history
        if any(st.session_state.export_filepath.values()):
            st.markdown("---")
            with st.expander("📋 Export History", expanded=False):
                for format, filepath in st.session_state.export_filepath.items():
                    if filepath:
                        st.write(f"- {format.upper()}: {filepath}") 