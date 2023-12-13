# Create virtual environment
python3 -m venv virtual

# Activate environment (UNIX)
(UNIX) source virtual/bin/activate
(WINDOWS) virtual/scripts/activate

# Install required packages
pip install -r requirements.txt


# Run Commands
python3 insert_data_pinecone.py


.env
PINECONE_API_KEY=fb1c9a46-68a8-4406-af1a-59705d655d2b
CSV_FILE_PATH=test.csv
EMBEDDING_COLUMN_NAME=question1
ID_COLUMN_NAME=test_id
INDEX_NAME=my-index
OPENAI_API_KEY=
