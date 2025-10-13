import chromadb
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import ollama

# --- 1. Configuration and Setup ---
load_dotenv()

# Get environment variables
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_COLLECTION_NAME = CHROMA_DATABASE
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL")

# Initialize embedding model
try:
    embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    print("Successfully loaded embedding model")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    exit(1)

def calculate_ranking_score(product):
    """Calculate a ranking score based on similarity, rating, and review count"""
    similarity = product['similarity_score']
    rating = float(product['stars']) if product['stars'] != '0' else 0
    reviews = int(product.get('reviews', '0'))
    
    # Normalize ratings and reviews (0-1 scale)
    normalized_rating = rating / 5.0  # assuming 5-star scale
    normalized_reviews = min(reviews / 1000.0, 1.0)  # cap at 1000 reviews
    
    # Weighted scoring (adjust weights as needed)
    similarity_weight = 0.4
    rating_weight = 0.3
    reviews_weight = 0.3
    
    ranking_score = (
        similarity * similarity_weight +
        normalized_rating * rating_weight +
        normalized_reviews * reviews_weight
    )
    
    return ranking_score

def get_product_recommendations(user_query: str, num_results: int = 3):
    """
    Generates an embedding for the user query and finds top 3 semantically similar products
    in ChromaDB using sentence transformers.
    """
    MIN_SIMILARITY = 0.7  # Only recommend products with similarity >= 0.7 (tune as needed)
    print(f"\nFinding top {num_results} products matching: '{user_query}'...")
    try:
        # Generate embedding for the user query
        query_embedding = embedding_model.embed_query(user_query)
        
        client = chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY
        )
        
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=10,  # Get more results, we'll filter by similarity
            include=['metadatas', 'documents', 'distances']
        )

        # Process results with enhanced ranking
        recommended_products = []
        if results and results['metadatas'] and results['metadatas'][0]:
            print(f"Found {len(results['metadatas'][0])} matches")
            
            for metadata, document, distance in zip(
                results['metadatas'][0],
                results['documents'][0],
                results['distances'][0]
            ):
                similarity_score = 1 - distance
                if similarity_score < MIN_SIMILARITY:
                    continue  # Skip low similarity
                product = {
                    'title': metadata.get('title', 'N/A'),
                    'price': metadata.get('price', '0.00'),
                    'stars': metadata.get('stars', '0'),
                    'reviews': metadata.get('reviewsCount', '0'),
                    'full_details': document,
                    'similarity_score': similarity_score
                }
                # Calculate ranking score
                product['ranking_score'] = calculate_ranking_score(product)
                recommended_products.append(product)
                print(f"Added product - Similarity: {similarity_score:.2%}, Rating: {product['stars']}, Reviews: {product['reviews']}")

        # Sort by ranking_score and return top N
        recommended_products.sort(key=lambda x: x['ranking_score'], reverse=True)
        recommended_products = recommended_products[:num_results]
        print(f"Final recommendations count: {len(recommended_products)}")
        return recommended_products

    except Exception as e:
        print(f"Error during product recommendation search: {e}")
        print(f"Full error details: {str(e)}")
        return []

def generate_recommendation_text(product, user_query):
    """Generate personalized recommendation text using llama3.2:3b"""
    prompt = f"""[INST] As a knowledgeable shopping assistant, create an engaging product recommendation:

User Search: "{user_query}"
Product Details:
- Name: {product['title']}
- Price: ${product['price']}
- Rating: {product['stars']} stars
- Reviews: {product['reviews']} customer reviews
- Match Relevance: {product['similarity_score']:.2%}

Create a response that:
1. Starts with "I found this excellent match:"
2. Mentions the product name
3. Describes 2-3 key features or benefits
4. Includes the price point
5. Mentions rating only if above 0 stars
6. Includes review count only if above 0
7. Uses a friendly, enthusiastic tone
8. Keeps to 2-3 sentences maximum

Format:
"I found this excellent match: [Product Name]. [Key features/benefits]. Available for $[price] [include rating/reviews if applicable]."

Your recommendation: [/INST]"""

    try:
        response = ollama.generate(
            model=OLLAMA_MODEL_NAME,
            prompt=prompt,
            temperature=0.5,
            max_tokens=300,
            top_p=0.5,
            top_k=40
        )
        return response['response'].strip()
    except Exception as e:
        fallback = f"I found this excellent match: {product['title']}. "
        fallback += f"This product matches your search criteria and is available for ${product['price']}. "
        if product['stars'] != '0':
            fallback += f"It has a rating of {product['stars']} stars. "
        if product['reviews'] != '0':
            fallback += f"({product['reviews']} customer reviews)"
        return fallback

def display_recommendations(products, user_query):
    """Display formatted product recommendations with AI-generated text"""
    if not products:
        print("\nNo products found matching your request.")
        return

    print("\n=== Personalized Recommendations ===")
    for i, product in enumerate(products, 1):
        print(f"\n📦 Recommendation #{i}")
        print("─" * 50)
        
        # Generate and display AI recommendation
        ai_recommendation = generate_recommendation_text(product, user_query)
        print(f"🤖 AI Assistant: {ai_recommendation}")
        
        # Display structured details
        print("Product Details:")
        print(f"   • Title: {product['title']}")
        print(f"   • Price: ${product['price']}")
        if product['stars'] != '0':
            print(f"   • Rating: {'⭐' * int(float(product['stars']))} ({product['stars']})")
        if product['reviews'] != '0':
            print(f"   • Reviews: {product['reviews']}")
        print(f"   • Relevance: {product['similarity_score']:.2%}")
        print(f"   • Overall Score: {product['ranking_score']:.2%}")
        
    print("\n" + "═"*50)

import streamlit as st

st.title("🛍️ AI Shopping Assistant")
st.write("Ask me about any products you're interested in!")

user_input = st.text_input("🔍 What are you looking for?", "")

if user_input:
    recommended_products = get_product_recommendations(user_input, num_results=3)
    if not recommended_products:
        st.warning("No products found matching your request.")
    else:
        st.subheader("Personalized Recommendations")
        for i, product in enumerate(recommended_products, 1):
            st.markdown(f"### 📦 Recommendation #{i}")
            ai_recommendation = generate_recommendation_text(product, user_input)
            st.info(f"🤖 AI Assistant: {ai_recommendation}")
            with st.expander("Product Details"):
                st.write(f"**Title:** {product['title']}")
                st.write(f"**Price:** ${product['price']}")
                if product['stars'] != '0':
                    st.write(f"**Rating:** {product['stars']} ⭐")
                if product['reviews'] != '0':
                    st.write(f"**Reviews:** {product['reviews']}")
                st.write(f"**Relevance:** {product['similarity_score']:.2%}")
                st.write(f"**Overall Score:** {product['ranking_score']:.2%}")
