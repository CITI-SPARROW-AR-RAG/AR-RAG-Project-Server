from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, Index, utility, connections
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
from fastapi import FastAPI, HTTPException

import os, logging, ollama, time, json, hashlib, secrets, string, datetime, uvicorn


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

@app.post("/admin/verify_password")
def verfiy_password(username, pass_input):
    with open(USERS_FILE, 'r') as f:
        users = json.load(f)

    stored_hash = users[username]["password_hash"]
    salt = users[username]["salt"]
    old_hash, _ = hash_password(pass_input, salt)

    if old_hash != stored_hash:
        return False
    else:
        return True

@app.put("/admin/change_password")
def change_password(username, new_password):
    new_hash, new_salt = hash_password(new_password)

    with open(USERS_FILE, 'r') as f:
        users = json.load(f)
    
    users[username]["password_hash"] = new_hash
    users[username]["salt"] = new_salt

    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# 
def create_testset():
    pass

# 
def get_testset_dataframe():
    pass

# 
def get_testsets_info():
    pass

#
def ragas_evaluation():
    pass

#
def get_evaluations_info():
    pass

#
def get_evaluation():
    pass

#
def delete_evaluation_file():
    pass

if __name__ == "__main__":
    import uvicorn
    create_initial_admin()
    uvicorn.run("milvus-db:app", host="0.0.0.0", port=8000, reload=True)