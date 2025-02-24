import pandas as pd
from pathlib import Path
import json
import sys

def validate_exports():
    """Validate export files structure and content"""
    export_dir = Path('data/exports')
    
    if not export_dir.exists():
        print("❌ Exports directory not found")
        sys.exit(1)
    
    # Validate CSV exports
    for csv_file in export_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file)
            required_cols = ['timestamp', 'market_status', 'drift_score']
            if not all(col in df.columns for col in required_cols):
                print(f"❌ Missing required columns in {csv_file}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error validating {csv_file}: {e}")
            sys.exit(1)
    
    # Validate JSON exports
    for json_file in export_dir.glob('*.json'):
        try:
            with open(json_file) as f:
                data = json.load(f)
            required_keys = ['timestamp', 'analysis', 'recommendations']
            if not all(key in data for key in required_keys):
                print(f"❌ Missing required keys in {json_file}")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Error validating {json_file}: {e}")
            sys.exit(1)
    
    print("✅ Export validation complete")

if __name__ == "__main__":
    validate_exports() 