import sys
import subprocess
import pkg_resources

def check_dependency(package):
    try:
        pkg_resources.require(package)
        return True
    except pkg_resources.DistributionNotFound:
        return False

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Diagnosing Dask installation...")

# First, uninstall existing dask-related packages
packages_to_remove = ['dask', 'distributed', 'fsspec', 'cloudpickle']
for package in packages_to_remove:
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", package])

# Install core dependencies first
core_deps = [
    "numpy>=1.26.0",
    "pandas>=2.2.0",
]

print("\nInstalling core dependencies...")
for dep in core_deps:
    if not check_dependency(dep):
        install_package(dep)

# Now install dask and its components in the correct order
dask_deps = [
    "cloudpickle>=3.0.0",
    "fsspec>=2024.2.0",
    "distributed>=2024.1.1",
    "dask[complete]>=2024.1.1"
]

print("\nInstalling Dask and dependencies...")
for dep in dask_deps:
    install_package(dep)

# Verify installation
print("\nVerifying installation...")
try:
    import dask.dataframe as dd
    print("✓ Dask DataFrame imported successfully")
    import distributed
    print("✓ Distributed imported successfully")
    print("\nDask installation complete and verified!")
except ImportError as e:
    print(f"Error: {e}")
    print("Please try running this script again or contact support.") 