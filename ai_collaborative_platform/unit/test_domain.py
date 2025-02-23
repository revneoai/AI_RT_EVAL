import pytest
from datetime import datetime
from domain.entities import DataRecord, Model

def test_data_record_creation():
    """Test data record entity creation"""
    record = DataRecord(
        id="test-1",
        source="test",
        content={"value": 1},
        created_at=datetime.now()
    )
    assert record.id == "test-1"
    assert record.source == "test"
