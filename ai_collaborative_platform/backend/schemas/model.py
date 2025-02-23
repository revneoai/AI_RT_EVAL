from pydantic import BaseModel, Field
from typing import List, Optional, Union
from enum import Enum

class ModelFramework(str, Enum):
    """Supported ML frameworks"""
    SKLEARN = "sklearn"
    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"

class ModelMetadata(BaseModel):
    """Metadata for model versioning and tracking"""
    name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Model version")
    framework: ModelFramework
    description: Optional[str] = None

class ModelPredictionRequest(BaseModel):
    """Schema for model prediction requests"""
    model_id: str = Field(..., description="ID of the model to use for prediction")
    features: List[dict] = Field(..., description="List of feature dictionaries")
    options: Optional[dict] = Field(default={}, description="Additional prediction options")

    class Config:
        schema_extra = {
            "example": {
                "model_id": "model-123",
                "features": [
                    {"feature1": 0.5, "feature2": 1.0},
                    {"feature1": 0.7, "feature2": 0.3}
                ],
                "options": {
                    "return_probability": True,
                    "threshold": 0.5
                }
            }
        }

class ModelPredictionResponse(BaseModel):
    """Schema for model prediction responses"""
    predictions: List[Union[float, str, dict]] = Field(..., description="Model predictions")
    model_version: str = Field(..., description="Version of the model used")
    prediction_time: float = Field(..., description="Time taken for prediction in seconds")
    confidence_scores: Optional[List[float]] = None

    class Config:
        schema_extra = {
            "example": {
                "predictions": [0.8, 0.2],
                "model_version": "1.0.0",
                "prediction_time": 0.0123,
                "confidence_scores": [0.95, 0.87]
            }
        }
