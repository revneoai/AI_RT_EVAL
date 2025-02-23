from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

class SQLAlchemyAdapter:
    """SQLAlchemy database adapter"""
    def __init__(self, session_factory: sessionmaker):
        self.session_factory = session_factory

    async def get_session(self) -> AsyncSession:
        return self.session_factory()
