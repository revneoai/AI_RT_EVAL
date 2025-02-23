from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

@dataclass
class DataRecord:
    """Core domain entity for data records"""
    id: str
    source: str
    content: dict
    created_at: datetime
    metadata: Optional[dict] = None

@dataclass
class Model:
    """Core domain entity for ML models"""
    id: str
    name: str
    version: str
    framework: str
    created_at: datetime
    metrics: Optional[dict] = None
