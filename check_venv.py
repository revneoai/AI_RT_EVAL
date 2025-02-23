import os
import sys
import subprocess
import platform

def is_venv_active():
    """Check if a virtual environment is active"""
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

def create_and_activate_venv():
    """Create and activate a virtual environment"""
    system = platform.system().lower()
    
    try:
        if not os.path.exists('venv'):
            print("Creating virtual environment...")
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        
        # Determine activation script path based on OS
        if system == 'windows':
            activate_script = os.path.join('venv', 'Scripts', 'activate')
            activate_cmd = f"call {activate_script}"
        else:  # Linux/Mac
            activate_script = os.path.join('venv', 'bin', 'activate')
            activate_cmd = f"source {activate_script}"
        
        print(f"\nTo activate the virtual environment, run:")
        print(f"{activate_cmd}")
        print("\nAfter activation, install requirements with:")
        print("pip install -r requirements.txt")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def main():
    if is_venv_active():
        print("Virtual environment is already active!")
        print("\nYou can proceed with:")
        print("pip install -r requirements.txt")
    else:
        print("No virtual environment is active.")
        create_and_activate_venv()
        print("\nAfter installing requirements, you can run:")
        print("python create_project_ad.py")

if __name__ == "__main__":
    main() 