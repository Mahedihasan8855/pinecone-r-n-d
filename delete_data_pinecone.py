import os
from dotenv import load_dotenv
import pinecone
import openai

# Load environment variables from .env
load_dotenv()

# Read environment variables
api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
openai_api_key = os.getenv('OPENAI_API_KEY')  # Your OpenAI API key from environment variables

openai.api_key = openai_api_key

# Initialize Pinecone connection
pinecone.init(api_key=api_key, environment="gcp-starter")
index = pinecone.Index(index_name)

def delete_data_from_pinecone(data_id):
    # Delete data from Pinecone
    delete_response = index.delete(ids=[data_id])

    print(f"Data with ID {data_id} deleted from Pinecone successfully.")

# Example usage:
data_id_to_delete = "LinkinPark_000_text"
delete_data_from_pinecone(data_id_to_delete)