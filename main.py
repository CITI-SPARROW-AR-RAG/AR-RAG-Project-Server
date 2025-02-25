from fastapi import FastAPI
from pydantic import BaseModel
import ollama
from pymilvus import MilvusClient
import uvicorn
import numpy as np

app = FastAPI()

# Initialize Milvus client
milvus_client = MilvusClient(uri="http://localhost:19530")
COLLECTION_NAME = "vanessa_rag_collection"

# Model configurations
EMBEDDING_MODEL = "nomic-embed-text:latest"
LLM_MODEL = "qwen2.5:14b"

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
    SYSTEM_PROMPT = """You are an AI assistant designed to support operators in the manufacturing environment. Your goal is to be precise, helpful, and responsive in assisting with tasks. Follow the user prompt instructions carefully."""
    USER_PROMPT = f"""
                You are a knowledgeable AI assistant designed to support operators in the manufacturing environment. Please follow these guidelines for responding:

                1. If the question relates to procedures or guidelines with clear steps, respond in a structured manner using numbered lists or bullet points (e.g., Step 1: ..., Step 2: ...). Ensure that each step is listed clearly and individually, separating each instruction for easy understanding.
                2. If the question is unclear or lacks sufficient details, kindly ask the user for clarification to provide an accurate response.
                3. If you do not have the information available in your database, state that you're unable to provide an answer at this time.
                4. After sending your initial response, send a follow-up message a few seconds later encouraging the user to ask if they have further inquiries.

                Use these context passages to answer the question. Include only information from the context:
                Context: {context}
                Question: {question}

                After your response, encourage the user to ask further questions if needed.
                """

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
    ]

def rerank_chunks(question_embedding, retrieved_chunks):
    """Rerank the chunks based on similarity to the question embedding"""
    similarities = []
    for chunk in retrieved_chunks:
        chunk_embedding = emb_text(chunk)  # Get embedding for each chunk
        similarity = np.dot(question_embedding, chunk_embedding)  # Cosine similarity
        similarities.append(similarity)
    
    # Sort chunks by similarity in descending order
    ranked_chunks = [x for _, x in sorted(zip(similarities, retrieved_chunks), reverse=True)]
    return ranked_chunks

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # 1. Create embedding for the question
        question_embedding = emb_text(request.question)
        
        # 2. Search in Milvus
        search_results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[question_embedding],
            limit=3,
            search_params={"metric_type": "IP", "params": {}},
            output_fields=["text"]
        )
        
        # 3. Format retrieved contexts
        retrieved_texts = [res["entity"]["text"] for res in search_results[0]]
        
        # 4. Rerank the retrieved contexts based on similarity to the question
        reranked_texts = rerank_chunks(question_embedding, retrieved_texts)
        
        # 5. Generate response using LLM
        context = "\n".join(reranked_texts)  # Use reranked contexts
        messages = format_prompt(context, request.question)
        response = ollama.chat(model=LLM_MODEL, messages=messages) 
        
        return QueryResponse(
            answer=response["message"]["content"],
            sources=reranked_texts  # Return the reranked sources
        )
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
