from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, Index, utility, connections
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from fastapi import FastAPI

import os, logging, ollama, time


# GET PROJECT BASE DIR
current_dir = os.getcwd()
path = Path(current_dir)
BASE_DIR = path.parent 

app = FastAPI()


# Set up logging configuration
log_filename = os.path.join(BASE_DIR, 'config', 'logs', 'admin', f'ADMIN_{time.strftime("%Y-%m-%d_%H-%M-%S")}.log')

logging.basicConfig(
    filename=log_filename,                      # Log to file
    format='%(asctime)s - %(message)s',         # Add timestamp to each log message
    level=logging.INFO                          # Log level
)

# Create a logger instance
logger = logging.getLogger(__name__)

# Log the timestamp once at the beginning
initial_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
logger.info(f"Start of logging session at: {initial_timestamp}")


# READ ENV FILE
load_dotenv(BASE_DIR / ".env")

MILVUS_URL = os.getenv("MILVUS_URL")
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")


# connecting to milvus database
try:
    connections.connect(alias="default", uri=MILVUS_URL)
    logger.info(f"✅ Successfully connected to Milvus at {MILVUS_URL}")
except Exception as e:
    logger.error(f"❌ Failed to connect to Milvus: {str(e)}")

#
def emb_text(text):
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

#
def get_embedding_dimension():
    test_embedding = emb_text("This is a test")
    embedding_dim = len(test_embedding)
    return embedding_dim

# create collection
def create_collection():
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        logger.info(f"Existing collection {COLLECTION_NAME} dropped!")

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=get_embedding_dimension())
    ]

    schema = CollectionSchema(fields, description="Documents embeddings collection")

    # Create new collection
    collection = Collection(name=COLLECTION_NAME, schema=schema, consistency_level="Strong")
    logger.info(f"✅ '{COLLECTION_NAME}' collection successfully created!")

    # Create index on vector field
    index_params = {
        "metric_type": MILVUS_METRIC_TYPE,
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }

    collection.create_index(field_name="vector", index_params=index_params)
    logger.info("✅ Index successfully created!")

    return collection

# Read the saved text chunks from the file
def load_chunks():
    text_chunks = []

    with open(os.path.join(BASE_DIR, 'data', 'processed', 'chunks_output.txt'), "r", encoding="utf-8") as file:
        chunk = ""
        for line in file:
            # Detect new chunk start
            if line.startswith("Chunk "):
                if chunk:
                    text_chunks.append(chunk.strip())  # Save previous chunk
                chunk = ""  # Start new chunk
            else:
                chunk += line  # Append line to chunk

        # Save the last chunk
        if chunk:
            text_chunks.append(chunk.strip())

    return text_chunks

# 
def add_file_to_vdb(collection):
    text_chunks = load_chunks()
    embedding_vectors = [emb_text(text_chunk) for text_chunk in text_chunks]

    # Correct list comprehension
    entities = [
        {
            "text": text,
            "vector": vector
        }
        for i, (text, vector) in enumerate(tqdm(zip(text_chunks, embedding_vectors), desc="Processing embeddings"))
    ]

    # Insert into Milvus
    insert_result = collection.insert(entities)
    collection.flush()
    collection.load()
    print(f"Data berhasil dimasukkan dengan total {len(text_chunks)} chunk.")

# 
def remove_file_from_vdb():
    pass

# 
def upload_file():
    pass