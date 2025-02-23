from fastapi import APIRouter
router = APIRouter()

@router.get("/health")
async def health_check():
    """Health check endpoint required by prompt"""
    return {"status": "healthy"}
