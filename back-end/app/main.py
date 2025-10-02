# app/main.py
from fastapi import FastAPI
from app.routers import rag  # Import your router

app = FastAPI(title="RAG Backend API", version="0.1.0")

# Include the RAG router
app.include_router(rag.router)

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG Backend API!"}