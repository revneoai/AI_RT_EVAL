from typing import Dict, Any
from abc import ABC, abstractmethod

class EventHandler(ABC):
    """Abstract event handler"""
    @abstractmethod
    async def handle(self, event: Dict[str, Any]):
        """Handle incoming event"""
        pass

class DataIngestionEventHandler(EventHandler):
    """Handle data ingestion events"""
    async def handle(self, event: Dict[str, Any]):
        # Implementation
        pass

class ModelTrainingEventHandler(EventHandler):
    """Handle model training events"""
    async def handle(self, event: Dict[str, Any]):
        # Implementation
        pass
