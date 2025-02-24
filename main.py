from fastapi import FastAPI
from ai_collaborative_platform.interfaces.api.routes import router

app = FastAPI(
    title="AI Collaborative Platform",
    description="Real-time AI evaluation for property and fintech data",
    version="1.0.0"
)

# Include our API routes
app.include_router(router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} 