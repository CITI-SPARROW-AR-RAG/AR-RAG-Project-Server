from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, Index, utility, connections
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
import numpy as np
from pydantic import BaseModel
import pandas as pd
from datetime import datetime

import os, logging, ollama, time, json, hashlib, secrets, string, uvicorn, uuid, requests, glob


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



FILES_DIR = Path(__file__).parent.parent / "data" / "raw"
FILES_INDEX = Path(__file__).parent.parent / "data" / "raw" / "files_index.json"

if not os.path.exists(FILES_INDEX):
    with open(FILES_INDEX, 'w') as f:
        json.dump({}, f)


def emb_text(text: str) -> list[float]:
    """Generate embeddings using Ollama"""
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    #logger.info('reponse: ', response)
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
        FieldSchema(name="file_uuid", dtype=DataType.VARCHAR, max_length=36, is_primary=True),
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

def load_pdf(path_to_pdf):
    pdf_loader = PyPDFLoader(path_to_pdf)
    return pdf_loader.load()

def chunking_pdf(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)  
    return text_splitter.split_documents(documents)

# add file to vector database
@app.post("/admin/add_file_to_vdb")
def add_file_to_vdb(file_id: str = Form(...), file_metadata: str = Form(...)):
    """
    1. load the file
    2. chunking pdf file
    3. embed, then insert each chunk to vector database
    4. update status in files_index.json
    """

    # Parse file_metadata if it was sent as a JSON string
    file_metadata = json.loads(file_metadata)

    document = load_pdf(file_metadata["path"])
    chunks = chunking_pdf(document)

    # chunks = load_chunks()
    vector = [emb_text(chunk.page_content) for chunk in chunks]
    
    entities = [
        {
            "file_uuid": file_id,
            "text": chunk.page_content,
            "vector": emb
        }
        for chunk, emb in tqdm(zip(chunks, vector), desc="Processing embeddings")
    ]

    collection = Collection(name=COLLECTION_NAME)
    collection.insert(entities)
    collection.flush()

    with open(FILES_INDEX, 'r') as f:
        files = json.load(f)
    
    files[file_id]["in_vector_db"] = True

    with open(FILES_INDEX, 'w') as f:
        json.dump(files, f, indent=4)

    return {"status": True, "message": "File added successfully"}

# 
@app.post("/admin/remove_file_from_vdb")
def remove_file_from_vdb(file_id: str = Form(...)):
    """
    1. Remove all vectors associated with the file_id from the vector database.
    2. Update the status in files_index.json.
    """
    # Connect to Milvus
    collection = Collection(name=COLLECTION_NAME)

    # Ensure collection exists and is loaded
    collection.load()

    # Delete vectors where file_uuid matches file_id
    deletion_expr = f'file_uuid == "{file_id}"'
    collection.delete(deletion_expr)
    collection.flush()

    # Update file index
    if os.path.exists(FILES_INDEX):
        with open(FILES_INDEX, 'r') as f:
            files = json.load(f)
        
        if file_id in files:
            files[file_id]["in_vector_db"] = False  # Mark as removed
            
            with open(FILES_INDEX, 'w') as f:
                json.dump(files, f, indent=4)

    return {"status": True, "message": "File removed successfully"}

# Read the saved text chunks from the file
# def load_chunks():
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
@app.post("/admin/upload_file")
async def upload_file(
    username: str = Form(...),
    uploaded_file: UploadFile = File(...),
    in_vector_db: bool = Form(False)
):
    """Save an uploaded file and record its metadata"""
    
    # Generate a unique ID for the file
    file_id = str(uuid.uuid4())
    
    # Extract file extension
    original_filename = uploaded_file.filename
    ext = os.path.splitext(original_filename)[1]
    
    # Create a unique filename with the original extension
    filename = f"{file_id}{ext}"
    file_path = os.path.join(FILES_DIR, filename)
    
    # Save the file
    with open(file_path, "wb") as f:
        f.write(await uploaded_file.read())  # Use `await` karena UploadFile bersifat async
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Record metadata
    if not os.path.exists(FILES_INDEX):
        files_index = {}
    else:
        with open(FILES_INDEX, 'r') as f:
            files_index = json.load(f)
    
    files_index[file_id] = {
        "original_filename": original_filename,
        "stored_filename": filename,
        "upload_time": str(datetime.datetime.now()),
        "uploader": username,
        "file_size_bytes": file_size,
        "file_type": uploaded_file.content_type,
        "in_vector_db": in_vector_db,
        "path": file_path
    }
    
    with open(FILES_INDEX, 'w') as f:
        json.dump(files_index, f, indent=4)
    
    return {"file_id": file_id, "metadata": files_index[file_id]}

@app.get("/admin/list_files")
def list_files():
    """Get a list of all uploaded files with their metadata"""
    if not os.path.exists(FILES_INDEX):
        return {}
    
    with open(FILES_INDEX, 'r') as f:
        files_index = json.load(f)
    
    return files_index

def get_file_metadata(file_id):
    """Get metadata for a specific file"""
    if not os.path.exists(FILES_INDEX):
        return None
    
    with open(FILES_INDEX, 'r') as f:
        files_index = json.load(f)
    
    return files_index.get(file_id)

@app.get("/admin/download_file")
def download_file(file_id: str):
    """Get file data for download"""
    file_metadata = get_file_metadata(file_id)  # Pastikan fungsi ini sudah ada
    
    if not file_metadata:
        return {"error": "File not found"}
    
    file_path = os.path.join(FILES_DIR, file_metadata["stored_filename"])
    
    if not os.path.exists(file_path):
        return {"error": "File data not found"}

    # Streaming file ke client
    return StreamingResponse(open(file_path, "rb"), media_type="application/octet-stream", headers={
        "Content-Disposition": f"attachment; filename={file_metadata['original_filename']}"
    })

@app.delete("/admin/delete_file")
def delete_file(file_id):
    """Delete a file and its metadata"""
    if not os.path.exists(FILES_INDEX):
        return {"status": False, "message": "Files index not found"}
    
    with open(FILES_INDEX, 'r') as f:
        files_index = json.load(f)
    
    if file_id not in files_index:
        return {"status": False, "message": "File not found"}
    
    # Get the file info
    file_info = files_index[file_id]
    file_path = os.path.join(FILES_DIR, file_info["stored_filename"])
    
    # Delete the file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Remove from index
    del files_index[file_id]
    
    # Update index file
    with open(FILES_INDEX, 'w') as f:
        json.dump(files_index, f, indent=4)
    
    return {"status": True, "message": "File deleted successfully"}

# Define the path to the users file
USERS_FILE = Path(__file__).parent.parent / "data" / "users" / "users.json"
# os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

def hash_password(password, salt=None):
    """Hash a password with a salt for secure storage"""
    if salt is None:
        # Generate a random salt
        alphabet = string.ascii_letters + string.digits
        salt = ''.join(secrets.choice(alphabet) for _ in range(16))
    
    # Combine password and salt and hash
    password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
    return password_hash, salt

@app.get("/admin/check_login")
def check_login(username, password):
    """Verify login credentials"""
    if not os.path.exists(USERS_FILE):
        # If no users file exists, create a default admin user for first-time setup
        create_initial_admin()
    
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    if username in users:
        stored_hash = users[username]['password_hash']
        salt = users[username]['salt']
        
        # Hash the provided password with the stored salt
        input_hash, _ = hash_password(password, salt)
        
        if input_hash == stored_hash:
            return True
    
    return False

@app.post("/admin/create_user")
def create_user(username, password, created_by):
    """Create a new admin user"""
    if not os.path.exists(USERS_FILE):
        users = {}
    else:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    
    # Check if username already exists
    if username in users:
        return False, "Username already exists"
    
    # Hash the password
    password_hash, salt = hash_password(password)
    
    # Add the new user
    users[username] = {
        "password_hash": password_hash,
        "salt": salt,
        "created_by": created_by,
        "created_at": str(datetime.datetime.now())
    }
    
    # Save the updated users file
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)
    
    return {"success": True, "message": "User created successfully"}

def create_initial_admin():
    """Create an initial admin user if no users exist"""
    # Default credentials - you should change these immediately after first login
    username = "admin"
    password = "admin123"  # This is insecure - change after first login
    
    password_hash, salt = hash_password(password)
    
    users = {
        username: {
            "password_hash": password_hash,
            "salt": salt,
            "created_by": "system",
            "created_at": str(datetime.datetime.now())
        }
    }
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    
    # Save the users file
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)
    
    return True

@app.get("/admin/get_users")
def get_users_file():
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    return users

@app.get("/admin/verify_password")
def verify_password(username: str, pass_input: str):
    """Verify if the provided password matches the stored password for a user."""
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)

    stored_hash = users[username]["password_hash"]
    stored_salt = users[username]["salt"]

    # Hash the provided password using the stored salt
    old_hash, _ = hash_password(pass_input, stored_salt)

    return {"verified": old_hash == stored_hash}

@app.put("/admin/change_password")
def change_password(username: str, new_password: str):
    new_hash, new_salt = hash_password(new_password)

    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    if username not in users:
        return {"success": False, "message": "User not found"}

    users[username]["password_hash"] = new_hash
    users[username]["salt"] = new_salt

    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)
    
    return {"success": True, "message": "Password changed successfully"}



# Define constants
EVALUATIONS_DIR = Path(__file__).parent.parent / "data" / "evaluation"
EVAL_INDEX = Path(__file__).parent.parent / "data" / "evaluation" / "evaluation_index.json"
TESTSET_DIR = Path(__file__).parent.parent / "data" / "testset_generation"


# Define Pydantic model to validate request body
class QueryRequest(BaseModel):
    queries: list[str]

@app.post("/admin/get_queries_response")
def get_queries_response(request: QueryRequest):
    API_URL = "http://localhost:8080/query"
    result = []

    for query in request.queries:
        payload = {"question": query}
        response = requests.post(API_URL, json=payload)

        try:
            response_data = response.json()
        except Exception:
            raise HTTPException(status_code=500, detail="Invalid response from query server")

        response_data["question"] = query
        result.append(response_data)

    return {"responses": result}

class TestsetRequest(BaseModel):
    num_of_test: int  # Define input validation

@app.post("/admin/create_testset_using_ragas")
def create_testset_using_ragas(request: TestsetRequest):
    """Create a testset and save as CSV"""

    num_of_test = request.num_of_test  # Get input from request

    # Generate dummy data
    dummy_data = {
        'question': [f"Question {i}" for i in range(num_of_test)],
        'reference': [f"Reference {i}" for i in range(num_of_test)],
        'retrieved_context': [f"Context {i}" for i in range(num_of_test)],
    }
    testset_df = pd.DataFrame(dummy_data)

    # Save CSV file
    file_name = datetime.now().strftime("testset_%Y%m%d_%H%M%S.csv")
    file_path = os.path.join(TESTSET_DIR, file_name)
    testset_df.to_csv(file_path, index=False)

    # Return JSON response
    return {
        "success": True,
        "message": "Testset data created successfully",
        "file_name": file_name,
        "data": testset_df.to_dict(orient="records")  # Convert DataFrame to list of dictionaries
    }

@app.get("/admin/testset_files")
def get_testset_files_info():
    """Retrieve testset file information from the server"""
    testset_files = glob.glob(os.path.join(TESTSET_DIR, "testset_*.csv"))

    if not testset_files:
        return {"success": False, "message": "No testset files found", "files": []}

    file_data = []
    for testset_file in testset_files:
        filename = os.path.basename(testset_file)
        datetime_str = filename[8:-4]  # Extract timestamp
        dt = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")  # Convert to datetime

        file_data.append({
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "filename": filename
        })

    # Sort files by newest first
    file_data.sort(key=lambda x: x["datetime"], reverse=True)

    return {"success": True, "message": "Testset files retrieved", "files": file_data}


@app.get("/download")
def download_testset(file: str = Query(..., description="Filename to download")):
    """Download a testset file"""
    file_path = os.path.join(TESTSET_DIR, file)
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=file, media_type="text/csv")
    return {"success": False, "message": "File not found"}

if __name__ == "__main__":
    create_initial_admin()
    create_collection()
    
    uvicorn.run("milvus-db:app", host="127.0.0.1", port=8000, reload=True)