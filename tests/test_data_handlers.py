import pytest
from ai_collaborative_platform.backend.core.data_handlers.data_loader import DataLoader
from ai_collaborative_platform.backend.core.data_handlers.large_data_loader import LargeDataLoader
import pandas as pd

def test_data_loader_initialization():
    loader = DataLoader()
    assert loader is not None
    assert loader.chunk_size > 0

def test_large_data_loader_initialization():
    loader = LargeDataLoader()
    assert loader is not None
    assert loader.chunk_size > 0

def test_memory_status():
    loader = DataLoader()
    status = loader.get_memory_status()
    assert 'available_gb' in status
    assert status['available_gb'] > 0

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'price': [100000, 200000, 300000],
        'location': ['Dubai', 'Abu Dhabi', 'Sharjah'],
        'type': ['Apartment', 'Villa', 'Apartment']
    })

def test_data_validation(sample_data):
    loader = DataLoader()
    assert loader.validate_data(sample_data) == True 