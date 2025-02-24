import os
import json

def create_file(path, content=''):
    """Create a file with optional content"""
    with open(path, 'w') as f:
        f.write(content)

def create_directory_structure():
    # Base structure
    structure = {
        'backend': {
            'api': {
                '__init__.py': '',
                'data_endpoints.py': 'from fastapi import APIRouter\n\nrouter = APIRouter()\n',
                'model_endpoints.py': 'from fastapi import APIRouter\n\nrouter = APIRouter()\n',
                'metrics_endpoints.py': 'from fastapi import APIRouter\n\nrouter = APIRouter()\n'
            },
            'core': {
                'data_pipeline': {
                    '__init__.py': '',
                    'quality_checks.py': '',
                    'cleaning.py': '',
                    'validation.py': '',
                    'drift_monitor.py': ''
                },
                'model_pipeline': {
                    '__init__.py': '',
                    'registry.py': '',
                    'architecture.py': '',
                    'training.py': '',
                    'evaluation.py': ''
                },
                'metrics': {
                    '__init__.py': '',
                    'data_metrics.py': '',
                    'model_metrics.py': '',
                    'combined_metrics.py': ''
                }
            },
            'database': {
                '__init__.py': '',
                'models.py': '',
                'connections.py': ''
            },
            'utils': {
                '__init__.py': '',
                'logging.py': '',
                'config.py': ''
            }
        },
        'frontend': {
            'src': {
                'components': {
                    'data': {
                        'DataQualityDashboard.tsx': '',
                        'DataValidationTools.tsx': '',
                        'DriftMonitoring.tsx': ''
                    },
                    'model': {
                        'ModelRegistry.tsx': '',
                        'PerformanceMetrics.tsx': '',
                        'ArchitectureVisualizer.tsx': ''
                    },
                    'shared': {
                        'Navigation.tsx': '',
                        'Charts.tsx': '',
                        'Alerts.tsx': ''
                    }
                },
                'pages': {
                    'Dashboard.tsx': '',
                    'DataStudio.tsx': '',
                    'ModelStudio.tsx': '',
                    'Analytics.tsx': ''
                },
                'utils': {
                    'api.ts': '',
                    'helpers.ts': ''
                }
            },
            'public': {},
            'package.json': json.dumps({
                "name": "ai-collaborative-platform",
                "version": "1.0.0",
                "dependencies": {
                    "react": "^18.2.0",
                    "typescript": "^4.9.0"
                }
            }, indent=2)
        },
        'tests': {
            'backend': {
                'test_data_pipeline.py': '',
                'test_model_pipeline.py': '',
                'test_metrics.py': ''
            },
            'frontend': {
                'component_tests': {}
            }
        },
        'docker': {
            'Dockerfile.backend': 'FROM python:3.9\nWORKDIR /app\n',
            'Dockerfile.frontend': 'FROM node:16\nWORKDIR /app\n',
            'docker-compose.yml': '''version: "3.8"
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
'''
        },
        'docs': {
            'api': {},
            'architecture': {},
            'user_guide': {}
        }
    }

    # Create root directory
    root_dir = 'ai_collaborative_platform'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    def create_structure(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            
            if isinstance(content, dict):
                # If it's a directory
                if not os.path.exists(path):
                    os.makedirs(path)
                create_structure(path, content)
            else:
                # If it's a file
                create_file(path, content)

    create_structure(root_dir, structure)
    print(f"Created application structure in {os.path.abspath(root_dir)}")

def main():
    create_directory_structure()
    print("Application structure created successfully!")
    print("\nNext steps:")
    print("1. Navigate to the project directory: cd ai_collaborative_platform")
    print("2. Initialize git repository: git init")
    print("3. Create virtual environment: python -m venv venv")
    print("4. Install backend dependencies: pip install fastapi uvicorn sqlalchemy")
    print("5. Install frontend dependencies: cd frontend && npm install")

if __name__ == "__main__":
    main()
import os
import json

def create_file(path, content=''):
    """Create a file with optional content"""
    with open(path, 'w') as f:
        f.write(content)

def create_directory_structure():
    # Base structure
    structure = {
        'backend': {
            'api': {
                '__init__.py': '',
                'data_endpoints.py': 'from fastapi import APIRouter\n\nrouter = APIRouter()\n',
                'model_endpoints.py': 'from fastapi import APIRouter\n\nrouter = APIRouter()\n',
                'metrics_endpoints.py': 'from fastapi import APIRouter\n\nrouter = APIRouter()\n'
            },
            'core': {
                'data_pipeline': {
                    '__init__.py': '',
                    'quality_checks.py': '',
                    'cleaning.py': '',
                    'validation.py': '',
                    'drift_monitor.py': ''
                },
                'model_pipeline': {
                    '__init__.py': '',
                    'registry.py': '',
                    'architecture.py': '',
                    'training.py': '',
                    'evaluation.py': ''
                },
                'metrics': {
                    '__init__.py': '',
                    'data_metrics.py': '',
                    'model_metrics.py': '',
                    'combined_metrics.py': ''
                }
            },
            'database': {
                '__init__.py': '',
                'models.py': '',
                'connections.py': ''
            },
            'utils': {
                '__init__.py': '',
                'logging.py': '',
                'config.py': ''
            }
        },
        'frontend': {
            'src': {
                'components': {
                    'data': {
                        'DataQualityDashboard.tsx': '',
                        'DataValidationTools.tsx': '',
                        'DriftMonitoring.tsx': ''
                    },
                    'model': {
                        'ModelRegistry.tsx': '',
                        'PerformanceMetrics.tsx': '',
                        'ArchitectureVisualizer.tsx': ''
                    },
                    'shared': {
                        'Navigation.tsx': '',
                        'Charts.tsx': '',
                        'Alerts.tsx': ''
                    }
                },
                'pages': {
                    'Dashboard.tsx': '',
                    'DataStudio.tsx': '',
                    'ModelStudio.tsx': '',
                    'Analytics.tsx': ''
                },
                'utils': {
                    'api.ts': '',
                    'helpers.ts': ''
                }
            },
            'public': {},
            'package.json': json.dumps({
                "name": "ai-collaborative-platform",
                "version": "1.0.0",
                "dependencies": {
                    "react": "^18.2.0",
                    "typescript": "^4.9.0"
                }
            }, indent=2)
        },
        'tests': {
            'backend': {
                'test_data_pipeline.py': '',
                'test_model_pipeline.py': '',
                'test_metrics.py': ''
            },
            'frontend': {
                'component_tests': {}
            }
        },
        'docker': {
            'Dockerfile.backend': 'FROM python:3.9\nWORKDIR /app\n',
            'Dockerfile.frontend': 'FROM node:16\nWORKDIR /app\n',
            'docker-compose.yml': '''version: "3.8"
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
  frontend:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "3000:3000"
'''
        },
        'docs': {
            'api': {},
            'architecture': {},
            'user_guide': {}
        }
    }

    # Create root directory
    root_dir = 'ai_collaborative_platform'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    def create_structure(base_path, structure):
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            
            if isinstance(content, dict):
                # If it's a directory
                if not os.path.exists(path):
                    os.makedirs(path)
                create_structure(path, content)
            else:
                # If it's a file
                create_file(path, content)

    create_structure(root_dir, structure)
    print(f"Created application structure in {os.path.abspath(root_dir)}")

def main():
    create_directory_structure()
    print("Application structure created successfully!")
    print("\nNext steps:")
    print("1. Navigate to the project directory: cd ai_collaborative_platform")
    print("2. Initialize git repository: git init")
    print("3. Create virtual environment: python -m venv venv")
    print("4. Install backend dependencies: pip install fastapi uvicorn sqlalchemy")
    print("5. Install frontend dependencies: cd frontend && npm install")

if __name__ == "__main__":
    main()
