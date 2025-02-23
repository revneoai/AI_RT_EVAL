from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class DataIngestionRequest(BaseModel):
    """Schema for data ingestion requests"""
    source: str = Field(..., description="Data source identifier")
    data: List[Dict[str, Any]] = Field(..., description="Data records to ingest")
    metadata: Optional[dict] = Field(default={}, description="Additional metadata")
    validation_rules: Optional[dict] = None

    @validator('data')
    def validate_data_not_empty(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        return v

    class Config:
        schema_extra = {
            "example": {
                "source": "customer_database",
                "data": [
                    {"id": 1, "feature1": 0.5, "feature2": "value"},
                    {"id": 2, "feature1": 0.7, "feature2": "other"}
                ],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "validation_rules": {
                    "feature1": {"type": "float", "range": [0, 1]},
                    "feature2": {"type": "string", "max_length": 50}
                }
            }
        }

class DataValidationResponse(BaseModel):
    """Schema for data validation responses"""
    is_valid: bool = Field(..., description="Overall validation status")
    validation_errors: List[dict] = Field(default=[], description="List of validation errors")
    metadata: Optional[dict] = None

    class Config:
        schema_extra = {
            "example": {
                "is_valid": False,
                "validation_errors": [
                    {
                        "record_id": 1,
                        "field": "feature1",
                        "error": "Value out of range [0, 1]"
                    }
                ],
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "record_count": 100
                }
            }
        }
