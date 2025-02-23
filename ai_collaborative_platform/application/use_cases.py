from typing import Optional
from ..domain.entities import DataRecord, Model
from ..domain.repositories import DataRepository, ModelRepository

class ProcessDataUseCase:
    """Use case for processing data records"""
    def __init__(self, data_repository: DataRepository):
        self.data_repository = data_repository

    async def execute(self, record: DataRecord) -> DataRecord:
        # Implementation
        return await self.data_repository.save(record)

class TrainModelUseCase:
    """Use case for training ML models"""
    def __init__(self, model_repository: ModelRepository):
        self.model_repository = model_repository

    async def execute(self, model: Model) -> Model:
        # Implementation
        return await self.model_repository.save(model)
