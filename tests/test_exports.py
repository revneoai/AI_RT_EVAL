import pytest
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

@pytest.fixture
def sample_analysis():
    return {
        'timestamp': datetime.now(),
        'market_trends': {
            'status': 'stable',
            'details': 'Market showing normal behavior'
        },
        'price_anomalies': {
            'count': 2,
            'details': ['Anomaly 1', 'Anomaly 2']
        },
        'drift_analysis': {
            'score': 0.05
        },
        'recommendations': [
            'Monitor price trends',
            'Check market stability'
        ]
    }

def test_export_directory_exists():
    export_dir = Path('data/exports')
    assert export_dir.exists(), "Export directory should exist"

def test_csv_export(sample_analysis):
    from scripts.validate_exports import validate_exports
    
    # Create test export
    export_dir = Path('data/exports')
    export_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame([{
        'timestamp': sample_analysis['timestamp'],
        'market_status': sample_analysis['market_trends']['status'],
        'drift_score': sample_analysis['drift_analysis']['score']
    }])
    
    test_file = export_dir / 'test_export.csv'
    df.to_csv(test_file, index=False)
    
    # Validate should not raise any errors
    validate_exports()
    
    # Cleanup
    test_file.unlink()

def test_json_export(sample_analysis):
    from scripts.validate_exports import validate_exports
    
    # Create test export
    export_dir = Path('data/exports')
    export_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = export_dir / 'test_export.json'
    with open(test_file, 'w') as f:
        json.dump(sample_analysis, f, default=str)
    
    # Validate should not raise any errors
    validate_exports()
    
    # Cleanup
    test_file.unlink() 