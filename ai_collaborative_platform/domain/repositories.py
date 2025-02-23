from abc import ABC, abstractmethod
from typing import List, Optional
from .entities import DataRecord, Model

class DataRepository(ABC):
    """Abstract repository for data records"""
    @abstractmethod
    async def save(self, record: DataRecord) -> DataRecord:
        pass

    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[DataRecord]:
        pass

    @abstractmethod
    async def find_all(self) -> List[DataRecord]:
        pass

class ModelRepository(ABC):
    """Abstract repository for ML models"""
    @abstractmethod
    async def save(self, model: Model) -> Model:
        pass

    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[Model]:
        pass

    @abstractmethod
    async def find_all(self, id: str) -> List[Model]:
        pass
