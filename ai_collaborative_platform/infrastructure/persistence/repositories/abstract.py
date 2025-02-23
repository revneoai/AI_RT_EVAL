from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Optional, List
from ...domain.entities import DataRecord, Model

T = TypeVar('T')

class Repository(Generic[T], ABC):
    """Abstract repository pattern implementation"""
    @abstractmethod
    async def save(self, entity: T) -> T:
        pass

    @abstractmethod
    async def find_by_id(self, id: str) -> Optional[T]:
        pass

    @abstractmethod
    async def find_all(self) -> List[T]:
        pass
