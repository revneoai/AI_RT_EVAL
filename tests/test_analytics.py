import pytest
from ai_collaborative_platform.backend.core.evaluation.drift_detector import DriftDetector
from ai_collaborative_platform.backend.core.evaluation.anomaly_detector import AnomalyDetector, AnomalyConfig
import numpy as np

def test_drift_detector_initialization():
    detector = DriftDetector(threshold=0.1)
    assert detector is not None
    assert detector.threshold == 0.1

def test_anomaly_detector_initialization():
    config = AnomalyConfig(z_score_threshold=3.0, window_size=1000)
    detector = AnomalyDetector(config=config)
    assert detector is not None
    assert detector.config.z_score_threshold == 3.0

@pytest.fixture
def sample_data():
    return np.random.normal(100, 15, 1000)

def test_drift_detection(sample_data):
    detector = DriftDetector(threshold=0.1)
    result = detector.check_drift(sample_data)
    assert 'drift_score' in result
    assert isinstance(result['drift_score'], float)

def test_anomaly_detection(sample_data):
    config = AnomalyConfig(z_score_threshold=3.0, window_size=1000)
    detector = AnomalyDetector(config=config)
    result = detector.check_anomaly(sample_data)
    assert 'anomalies' in result
    assert isinstance(result['anomalies'], list) 