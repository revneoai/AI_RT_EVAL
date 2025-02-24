from pathlib import Path
import shutil
import sys

def sync_data_dirs():
    """Synchronize data directories and maintain consistency"""
    base_dir = Path('data')
    dirs = ['uploads', 'exports', 'baseline']
    
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        
        # Ensure directory exists
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Clean old temporary files
        for temp_file in dir_path.glob('*.tmp'):
            temp_file.unlink()
        
        # Create .gitkeep to maintain directory structure
        gitkeep = dir_path / '.gitkeep'
        gitkeep.touch()
    
    print("✅ Directory synchronization complete")

if __name__ == "__main__":
    sync_data_dirs() 