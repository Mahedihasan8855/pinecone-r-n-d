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

def embed_search_text(search_text):
    response = openai.Completion.create(
        engine="text-embedding-ada-002",
        prompt=search_text,
        max_tokens=100  # Retrieve the full text without any additional tokens
    )
    try:
        # Attempt to extract the embedding from the response
        print(response)
        embedding = response['choices'][0]['text'].strip()
        return embedding
    except (KeyError, IndexError):
        # Handle the case where the expected structure is not found
        print("Error: Unable to extract embedding from the OpenAI API response.")
        return None

def query_pinecone(embedded_search_text, top_k=3):
    # Query Pinecone for similar vectors
    query_result = index.query(
        vector=embedded_search_text,
        filter={
            "is_active": True,
            "is_trusted": True,
            "partner_id": 1,
            },
        top_k=top_k,
        include_values=True,
        include_metadata=True)
    
    # Extract results
    results = query_result['matches']

    return results

def prioritize_answers(user_question, answers):
    # Calculate embeddings for user question and answers
    question_embedding = embed_search_text(user_question)
    answer_embeddings = [embed_search_text(answer['data']['text']) for answer in answers]

    # Calculate cosine similarity between the question and each answer
    similarity_scores = [0.0]  # Placeholder, as OpenAI GPT doesn't provide similarity scores directly

    # Combine answers with their similarity scores
    answers_with_scores = list(zip(answers, similarity_scores))

    # Sort answers by similarity score in descending order
    sorted_answers = sorted(answers_with_scores, key=lambda x: x[1], reverse=True)

    # Return the sorted answers
    return [answer[0] for answer in sorted_answers]

search_text_to_embed = "Metallica"
embedded_search_text = embed_search_text(search_text_to_embed)
query_results = query_pinecone(embedded_search_text)

# Extract relevant information from Pinecone results
answers = [
    {
        'data': {
            'text': result.metadata['text'],  # Assuming your text is stored in the metadata
            'tags': result.metadata['tags'],  # Assuming your tags are stored in the metadata
        }
    }
    for result in query_results
]

# Prioritize and sort the answers based on semantic similarity
sorted_answers = prioritize_answers(search_text_to_embed, answers)

# Print the sorted answers
for idx, answer in enumerate(sorted_answers, start=1):
    print(f"{idx}. Answer: {answer['data']['text']}, Tags: {answer['data']['tags']}")
