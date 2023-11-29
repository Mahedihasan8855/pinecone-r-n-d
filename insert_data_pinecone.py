import os
from dotenv import load_dotenv
import pinecone
import openai
import datetime

# Load environment variables from .env
load_dotenv()

# Read environment variables
api_key = os.getenv('PINECONE_API_KEY')
index_name = os.getenv('INDEX_NAME')
openai_api_key = os.getenv('OPENAI_API_KEY')  # Your OpenAI API key from environment variables
env = os.getenv('ENVIRONMENT')
openai.api_key = openai_api_key

# Initialize Pinecone connection
pinecone.init(api_key=api_key, environment= env)
index = pinecone.Index(index_name)

def embed_text_chunk(text_to_embed):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text_to_embed
    )
    
    embedding = response["data"][0]["embedding"]
    
    return embedding

def insert_data_into_pinecone(data_id,data):
    # Extract data fields
    is_active = data.get("is_active", False)
    data_id = data_id
    text_chunk = data.get("text_chunk", "")

    # Embed text_chunk using OpenAI
    content_vector = embed_text_chunk(text_chunk)

    # Prepare data for insertion
    vector = content_vector
    metadata = {
        "filename":data.get("filename"),
        "is_active": is_active,
        "id": data_id,
        "text_chunk": text_chunk,
        "file_chunk_index": data.get("file_chunk_index", 0),
        "partner_id": data.get("partner_id", ""),
        "persona_id": data.get("persona_id", ""),
        "tag": data.get("tag", ""),
        "user_id": data.get("user_id", ""),
        "creator": data.get("creator", ""),
        "priority": data.get("priority", 0),
        "is_trusted": data.get("is_trusted", False),
        "has_image": data.get("has_image", False),
        "create_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Upsert data into Pinecone
    upsert_response = index.upsert(
        vectors=[(data_id, vector, metadata)]
    )

    print(f"Data with ID {data_id} loaded into Pinecone successfully.")

# Example usage:
sample_data = {
    "filename": "Inception.pdf",
    "text_chunk": "Inception is a 2010 science fiction action film[4][5][6] written and directed by Christopher Nolan, who also produced the film with Emma Thomas, his wife. The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious of his targets. He is offered a chance to have his criminal history erased, as payment for the implantation of another person's idea into a target's subconscious.[7] The ensemble cast includes Ken Watanabe, Joseph Gordon-Levitt, Marion Cotillard, Elliot Page,[a] Tom Hardy, Cillian Murphy, Tom Berenger, Dileep Rao and Michael Caine.",
    "combined_chunk": "combined data <p>....</p>",
    "file_chunk_index": 1,
    "tag": "movie",
    "partner_id": 1,
    "persona_id": 1,
    "user_id": 4,
    "creator": "John",
    "is_active": True,
    "priority": 1,
    "is_trusted": True,
    "has_image": False
}

insert_data_into_pinecone("Inception_0001_text",sample_data)
