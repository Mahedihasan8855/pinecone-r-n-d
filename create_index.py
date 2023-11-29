import os
from dotenv import load_dotenv
import pinecone

# Load environment variables from .env
load_dotenv()

# Read environment variables
api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
env = os.getenv('ENVIRONMENT')
pinecone.init(api_key=api_key, environment= env)
metadata_config = {
    "indexed": ["color"]
}
if index_name in pinecone.list_indexes():
    print(f"Index '{index_name}' already exists. Skipping index creation.")
else:
    pinecone.create_index(
        index_name,
        dimension=1536,
        metadata_config=metadata_config)
    print(f"Index '{index_name}' created successfully.")


