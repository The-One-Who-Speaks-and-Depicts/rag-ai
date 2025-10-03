# app/routers/rag.py
from fastapi import APIRouter, HTTPException
from app.chroma_client import collection
import openai  # We still use the openai package, but configured for DeepSeek
import os
from dotenv import load_dotenv

load_dotenv()

# Configure the OpenAI client to use DeepSeek's API
openai.api_key = os.getenv("DEEPSEEK_API_KEY")
openai.api_base = "https://api.deepseek.com/v1"  # DeepSeek's API endpoint

router = APIRouter(prefix="/rag", tags=["RAG"])

@router.post("/query")
async def query_rag_system(user_query: str):
    """
    Endpoint to handle a user query, search ChromaDB, and generate an answer using DeepSeek.
    """
    
    # 1. Search for relevant context in ChromaDB
    try:
        results = collection.query(
            query_texts=[user_query],
            n_results=3  # Number of context chunks to retrieve
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
    
    # 3. Call the DeepSeek API
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",  # Use the appropriate DeepSeek model
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate answers based on the given context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for factual answers
            stream=False  # Set to True if you want streaming responses
        )
        answer = response.choices[0].message['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DeepSeek API error: {str(e)}")

    # 4. Return the answer
    return {
        "answer": answer, 
        "context_snippets": results['documents'][0],
        "model": "deepseek-chat"
    }