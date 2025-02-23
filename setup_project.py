import os
from create_project_ad import ProjectGenerator

def setup_project():
    """Setup the initial project structure"""
    # Create project generator
    generator = ProjectGenerator()
    
    # Create all project files
    generator.create_project()
    
    # Create main.py if it doesn't exist
    if not os.path.exists('main.py'):
        with open('main.py', 'w') as f:
            f.write('''from fastapi import FastAPI
from backend.interfaces.api.routes import router

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
''')

if __name__ == "__main__":
    setup_project() 