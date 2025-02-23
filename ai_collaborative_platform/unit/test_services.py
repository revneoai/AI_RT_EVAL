import pytest
from unittest.mock import Mock
from application.services import DataService

@pytest.fixture
def mock_repository():
    return Mock()

def test_data_service(mock_repository):
    """Test data service with mocked repository"""
    service = DataService(repository=mock_repository)
    mock_repository.find_by_id.return_value = {"id": "test"}
    result = service.get_data("test")
    assert result["id"] == "test"
