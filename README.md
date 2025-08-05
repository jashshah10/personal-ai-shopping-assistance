# AI Shopping Assistant

This project is a command-line interface (CLI) AI shopping assistant that provides personalized product recommendations based on user queries. It leverages a vector database (ChromaDB) for efficient semantic search and a local large language model (Ollama) to generate human-like recommendation text.

## Features

- **Natural Language Queries**: Find products by describing what you're looking for.
- **Semantic Search**: Uses sentence transformers to understand the meaning behind your query and find the most relevant products.
- **Personalized Recommendations**: Generates unique and helpful recommendation text for each product using a local LLM.
- **Data-driven Ranking**: Ranks products based on a combination of query similarity, user ratings, and review counts.

## Tech Stack

- **Python**: Core programming language.
- **ChromaDB**: Vector database for storing and querying product embeddings.
- **Sentence-Transformers**: For creating vector embeddings of product information and user queries.
- **Ollama**: Runs local large language models for generating recommendation text.
- **Pandas**: For data manipulation and loading product information from Excel files.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


4.  **Set up the environment file:**
    Create a file named `.env` in the project root and add the following environment variables. 

    ### Example `.env` file
    ```env
    # -- Chroma Cloud Settings --
    # Replace with your actual Chroma Cloud details.
    CHROMA_TENANT="your-chroma-tenant"
    CHROMA_DATABASE="your-chroma-database-name"
    CHROMA_API_KEY="your-chroma-api-key"

    # -- Ollama Settings --
    OLLAMA_MODEL_NAME="llama3.2:3b"  # Name of the Ollama model to use

    # -- Data and Model Settings --
    EXCEL_FILE_PATH="products_cleaned_1.xlsx"  # Path to your Excel file containing product data
    SENTENCE_TRANSFORMER_MODEL="multi-qa-MiniLM-L6-cos-v1"  # Model for sentence embeddings
    ```

## Data Schema

The project expects an Excel file (`.xlsx`) with your product data. The file should contain the following columns:

- `title`: The name or title of the product.
- `brand`: The brand of the product.
- `description`: A detailed description of the product.
- `price/value`: The price of the product.
- `stars`: The average user rating for the product (e.g., out of 5).
- `reviewsCount`: The total number of reviews for the product.

## Usage

1.  **Populate the Database**:
    Before running the main application, you need to process your product data and store it in ChromaDB. Run the `create_embeddings.py` script:
    ```bash
    python create_embeddings.py
    ```
    This script will read the product data from the Excel file specified in your `.env`, create vector embeddings, and upload them to your ChromaDB collection.

2.  **Run the AI Shopping Assistant**:
    Once the database is populated, you can start the interactive shopping assistant:
    ```bash
    python app.py
    ```
    You can then type your product queries directly into the terminal.

## File Descriptions

- `app.py`: The main application file for the interactive CLI shopping assistant.
- `create_embeddings.py`: A script to read product data, generate embeddings, and populate the ChromaDB database.
