from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI Service starting up...")
    yield
    print("AI Service shutting down...")

app = FastAPI(
    title="Learning Hub AI Service",
    description="AI/LLM Services for Multimodal Learning Hub",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Learning Hub AI Service", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "learning-hub-ai"}

@app.get("/ready")
def readiness_check():
    return {"ready": True}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)