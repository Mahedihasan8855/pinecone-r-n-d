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

openai.api_key = openai_api_key

# Initialize Pinecone connection
pinecone.init(api_key=api_key, environment="gcp-starter")
index = pinecone.Index(index_name)

def embed_text_chunk(text_to_embed):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text_to_embed
    )
    
    embedding = response["data"][0]["embedding"]
    
    return embedding

def upsert_data_into_pinecone(data_id, data):
    content = data.get("content", "")
    text_chunk = data.get("text_chunk", "")

    # Embed text_chunk using OpenAI
    content_vector = embed_text_chunk(text_chunk)

    # Prepare data for insertion or update
    vector = content_vector
    metadata = {
        "filename":data.get("filename"),
        "is_active":  data.get("is_active", False),
        "id": data_id,
        "content": content,
        "text_chunk": text_chunk,
        "file_chunk_index": data.get("file_chunk_index", 0),
        "partner_id": data.get("partner_id", ""),
        "persona_id": data.get("persona_id", ""),
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

    print(f"Data with ID {data_id} upserted into Pinecone successfully.")

# Example usage:
sample_data = {
    "filename": "Linkin Park --u.pdf",
    "text_chunk": "Linkin Park is an American rock band from Agoura Hills, California. The band's lineup comprises vocalist/rhythm guitarist/keyboardist Mike Shinoda, lead guitarist Brad Delson, bassist Dave Farrell, DJ/turntablist Joe Hahn and drummer Rob Bourdon, with vocalist Chester Bennington also part of the band until his death",
    "combined_chunk": "combined data <p>....</p>",
    "file_chunk_index": 1,
    "partner_id": 1,
    "persona_id": 1,
    "user_id": 4,
    "creator": "John",
    "is_active": True,
    "priority": 1,
    "is_trusted": True,
    "has_image": False
}

data_id = "LinkinPark_000_text"
upsert_data_into_pinecone(data_id, sample_data)
