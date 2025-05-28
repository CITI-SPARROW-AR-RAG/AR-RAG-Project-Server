from fastapi import FastAPI
from pydantic import BaseModel
import ollama
from pymilvus import MilvusClient
import uvicorn
import numpy as np
from dotenv import load_dotenv
import os
import logging
import time

import pandas as pd
from datetime import datetime
import csv




app = FastAPI()


# Set up logging configuration
logging.basicConfig(
    filename='RAG.log',                         # Log to file
    format='%(asctime)s - %(message)s',         # Add timestamp to each log message
    level=logging.INFO                          # Log level
)

# Create a logger instance
logger = logging.getLogger(__name__)

# Log the timestamp once at the beginning
initial_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"Start of logging session at: {initial_timestamp}")


# New function to save question and response to CSV
def save_to_csv(question, answer):
    csv_file = 'rag_qa_tracking.csv'
    file_exists = os.path.isfile(csv_file)
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Write to CSV
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['timestamp', 'question', 'answer'])
        writer.writerow([timestamp, question, answer])


# load .env variable
load_dotenv()

MILVUS_URL = os.getenv("MILVUS_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")
SERVER_HOST_URL = os.getenv("SERVER_HOST_URL")
SERVER_HOST_PORT = int(os.getenv("SERVER_HOST_PORT"))

#logging.info("MILVUS_URL: %s, Collection: %s", MILVUS_URL, COLLECTION_NAME)

# Initialize Milvus client
milvus_client = MilvusClient(uri=MILVUS_URL)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

def emb_text(text: str) -> list[float]:
    """Generate embeddings using Ollama"""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    #logger.info('reponse: ', response)
    return response["embedding"]


#User prompt and system prompt adapted for testing - please see Editor for original version

def format_prompt(context: str, question: str) -> list[dict]:
    """Format the prompt for the LLM"""
    SYSTEM_PROMPT = """You are an AI assistant supporting operators in a manufacturing environment. Your name is SPARROW.
                Your behavior must follow these principles:
                - Be precise, helpful, and responsive in all answers.
                - When a user asks about procedures, respond in clear, step-by-step instructions using numbered or bulleted lists.
                - If a question lacks sufficient detail, ask the user to clarify before answering.
                - Only use the information provided in the context. Do not guess or add external knowledge.
                - End each response by encouraging the user to ask additional questions.

                Maintain a professional but approachable tone at all times."""
    
    USER_PROMPT = f""" You are a knowledgeable AI assistant supporting operators in a manufacturing environment. Your name is SPARROW.
                Follow these detailed instructions to answer the user's question.

                Instructions:
                1. If the question involves a task, procedure, or guideline, format your answer as a step-by-step list using numbered or bullet points.
                2. Separate each step clearly for ease of understanding.
                3. If the question is unclear, missing context, or ambiguous, ask the user to clarify before attempting an answer.
                4. Only use the information provided in the context below. Do not guess or invent information.
                5. End your response with a polite prompt encouraging further questions. For example: "Let me know if you'd like help with anything else."

                Context:
                {context}

                Question:
                {question}
                """
    logging.info(f"User Prompt: {USER_PROMPT}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT}
            
    ]

def rerank_chunks(question_embedding, retrieved_chunks):
    """Rerank the chunks based on similarity to the question embedding"""
    similarities = []
    logging.info(f"retrieved chunks: {retrieved_chunks} ")
    for chunk in retrieved_chunks:
        chunk_embedding = emb_text(chunk)  # Get embedding for each chunk
        similarity = np.dot(question_embedding, chunk_embedding)  # Cosine similarity
        similarities.append(similarity)
    
    # Sort chunks by similarity in descending order
    ranked_chunks = [x for _, x in sorted(zip(similarities, retrieved_chunks), reverse=True)]
    logging.info(f"ranked: {ranked_chunks}")
    return ranked_chunks

@app.post("/query")
async def query_rag(request: QueryRequest):
    logging.info(f"question : {request.question}")
    try:
        # 1. Create embedding for the question
        question_embedding = emb_text(request.question)
        
        # 2. Search in Milvus
        search_results = milvus_client.search(
            collection_name=COLLECTION_NAME,
            data=[question_embedding],
            limit=2,
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
        
        logging.info(f"LLM_Model: {LLM_MODEL}")
        logging.info(f"LLM response: {response['message']['content']}")
#print response to console
        print(response["message"])


        return QueryResponse(   
            answer=response["message"]["content"],
            sources=reranked_texts  # Return the reranked sources
        )
    
         

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST_URL, port=SERVER_HOST_PORT)
