import pandas as pd
import chromadb
import os
from dotenv import load_dotenv
from langchain.embeddings import SentenceTransformerEmbeddings

# --- 1. Configuration and Setup ---
# Load environment variables from .env file
load_dotenv()

# Get ChromaDB credentials from environment
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
EXCEL_FILE_PATH = os.getenv("EXCEL_FILE_PATH")

# Use the database name for the collection name, as it's a good practice
CHROMA_COLLECTION_NAME = CHROMA_DATABASE

# Initialize the sentence transformer model
MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL")
embedding_model = SentenceTransformerEmbeddings(model_name=MODEL_NAME)

# --- 3. Load Data from Excel ---
try:
    # Reading from an Excel file. Requires 'openpyxl' to be installed.
    df = pd.read_excel(EXCEL_FILE_PATH)
    print(f"Successfully loaded {len(df)} products from '{EXCEL_FILE_PATH}'.")
except FileNotFoundError:
    print(f"Error: The file '{EXCEL_FILE_PATH}' was not found.")
    exit()
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit()

# --- 4. Initialize ChromaDB Cloud Client ---
try:
    print("Connecting to Chroma Cloud...")
    client = chromadb.CloudClient(
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
        api_key=CHROMA_API_KEY
    )
    print("Successfully connected to Chroma Cloud.")
except Exception as e:
    print(f"Error initializing ChromaDB Cloud client: {e}")
    exit()

# --- 5. Get or Create Chroma Collection ---
try:
    # Use the embedding model directly as it's already a LangChain embedding function
    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model
    )
    print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' created successfully.")
except Exception as e:
    print(f"Error getting/creating ChromaDB collection: {e}")
    exit()

# --- 6. Prepare and Upsert Data into ChromaDB ---
print("Preparing data for ChromaDB...")
try:
    # Prepare lists for ChromaDB
    # Generate unique IDs based on index since product_id is no longer available
    product_ids = df.index.astype(str).tolist()

    # Create rich document text combining multiple fields
    df['document_for_embedding'] = df.apply(
        lambda row: f"Product: {row.get('title', '')} - "
                   f"Brand: {row.get('brand', '')} - "
                   f"{row.get('description', '')} - "
                   f"Price: ${row.get('price/value', 0.00)} - "
                   f"Rating: {row.get('stars', 0)} stars - "
                   f"Reviews: {row.get('reviewsCount', 0)}", 
        axis=1
    )
    documents = df['document_for_embedding'].tolist()

    # Create metadata dictionary with only essential fields and default values
    metadatas = df.apply(
        lambda row: {
            'title': str(row.get('title', '')),
            'price': float(row.get('price/value', 0.00)),
            'stars': int(row.get('stars', 0))
        }, 
        axis=1
    ).tolist()

    # Upsert data to ChromaDB in batches
    BATCH_SIZE = 50
    total_docs = len(product_ids)

    if total_docs > 0:
        print(f"Starting to upsert {total_docs} documents in batches of {BATCH_SIZE}...")
        for i in range(0, total_docs, BATCH_SIZE):
            try:
                end_i = min(i + BATCH_SIZE, total_docs)
                print(f"Processing batch {i//BATCH_SIZE + 1}/{(total_docs + BATCH_SIZE - 1)//BATCH_SIZE}")
                
                batch_ids = product_ids[i:end_i]
                batch_documents = documents[i:end_i]
                batch_metadatas = metadatas[i:end_i]

                print(f"  ... generating embeddings for documents {i+1} to {end_i}")
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                print(f"  ... completed batch {i//BATCH_SIZE + 1}")
            except Exception as batch_error:
                print(f"Error in batch {i//BATCH_SIZE + 1}: {batch_error}")
                print(f"Skipping to next batch...")
                continue
        
        print(f"Successfully completed processing {total_docs} documents.")
    else:
        print("No products found in the CSV file to process.")

except Exception as e:
    print(f"Error uploading documents to ChromaDB: {e}")
    print(f"Full error details: {str(e)}")
    exit()

print(f"Total documents in ChromaDB collection '{CHROMA_COLLECTION_NAME}': {collection.count()}")