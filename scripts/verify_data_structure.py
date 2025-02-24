from pathlib import Path
import sys

def verify_data_structure():
    """Verify the required data directories exist and have correct permissions"""
    required_dirs = [
        'data/uploads',
        'data/exports',
        'data/baseline'
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True)
            print(f"Created missing directory: {dir_path}")
        
        # Verify permissions
        try:
            test_file = path / '.test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            print(f"Permission error in {dir_path}: {e}")
            sys.exit(1)
    
    print("✅ Data structure verification complete")

if __name__ == "__main__":
    verify_data_structure() 