# app/routers/rag.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.chroma_client import collection
from openai import OpenAI  # Import the new client
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / '.env')

# Initialize the new OpenAI client configured for DeepSeek
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"  # DeepSeek's API endpoint
)

router = APIRouter(prefix="/rag", tags=["RAG"])

# Define the request model
class QueryRequest(BaseModel):
    user_query: str

@router.post("/query")
async def query_rag_system(request: QueryRequest):
    """
    Endpoint to handle a user query, search ChromaDB, and generate an answer using DeepSeek.
    """
    
    # Extract the query from the request body
    user_query = request.user_query
    
    # 1. Search for relevant context in ChromaDB
    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=3
        )
        context = "\n\n".join(results['documents'][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # 2. Construct the augmented prompt
    prompt = f"""Use the following context to answer the user's question. If the answer is not in the context, say you don't know.

Context:
{context}

Question: {user_query}

Answer:"""
    
    # 3. Call the DeepSeek API using the new SDK
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            stream=False
        )
        answer = response.choices[0].message.content  # Note: .content instead of ['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek API error: {str(e)}")

    # 4. Return the answer
    return {
        "answer": answer, 
        "context_snippets": results['documents'][0],
        "model": "deepseek-chat"
    }