from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass(frozen=True)
class ModelMetrics:
    """Value object for model metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    created_at: datetime = datetime.now()

@dataclass(frozen=True)
class DataQualityMetrics:
    """Value object for data quality metrics"""
    completeness: float
    accuracy: float
    consistency: float
    timestamp: datetime = datetime.now()
