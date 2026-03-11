"""
Transport Service Launcher

This script runs the FastAPI application.
For development, use: uvicorn app.main:app --reload --port 8001
"""

from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

