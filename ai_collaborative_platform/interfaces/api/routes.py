from fastapi import APIRouter, Depends
from typing import List, Dict, Any
from ai_collaborative_platform.backend.core.evaluation.drift_detector import DriftDetector
from ai_collaborative_platform.backend.core.evaluation.anomaly_detector import AnomalyDetector
from ai_collaborative_platform.infrastructure.ai.evaluation.data_generator import DataGenerator

router = APIRouter()

@router.post("/evaluate/drift")
async def check_drift(data: List[float]):
    """Check for distribution drift in data"""
    detector = DriftDetector()
    detector.set_baseline(data[:500])  # Use first 500 points as baseline
    return detector.check_drift(data[500:])

@router.post("/evaluate/anomaly")
async def check_anomaly(value: float):
    """Check for anomalies in real-time"""
    detector = AnomalyDetector()
    return detector.check_anomaly(value)

@router.get("/generate/sample")
async def generate_sample(data_type: str = "property", size: int = 1000):
    """Generate synthetic data for testing"""
    generator = DataGenerator(data_type)
    return {"data": generator.generate_baseline(size)}
