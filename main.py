from fastapi import FastAPI
from pydantic import BaseModel
import ollama
from pymilvus import MilvusClient
import uvicorn
import os
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

# Initialize Milvus client
milvus_client = MilvusClient(uri=f"http://{os.getenv('MILVUS_CLIENT_URL')}:{os.getenv('MILVUS_CLIENT_PORT')}")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# Model configurations
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

def emb_text(text: str) -> list[float]:
    """Generate embeddings using Ollama"""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

def format_prompt(context: str, question: str) -> list[dict]:
    """Format the prompt for the LLM"""
    SYSTEM_PROMPT = "You are an AI assistant. Answer questions based on the provided context."
    USER_PROMPT = f"""
    Use these context passages to answer the question. Include only information from the context.
    Context: {context}
    Question: {question}
    """
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # 1. Create embedding for the question
        question_embedding = emb_text(request.question)
        
        search_results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[question_embedding],
            limit=3,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"]
        )
        
        # 3. Format retrieved contexts
        retrieved_texts = [res["entity"]["text"] for res in search_results[0]]
        context = "\n".join(retrieved_texts)
        
        # 4. Generate response using LLM
        messages = format_prompt(context, request.question)
        response = ollama.chat(model=LLM_MODEL, messages=messages)
        
        return QueryResponse(
            answer=response["message"]["content"],
            sources=retrieved_texts
        )
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)