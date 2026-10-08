import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_data(filepath="data/extracted_messages.json"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run previous phases.")
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    print("Loading messages...")
    messages = load_data()
    
    # We only embed the actual text of the message for semantic search
    texts = [msg["text"] for msg in messages]
    
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    # all-MiniLM-L6-v2 is a small, fast, and effective embedding model for general text
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"Generating embeddings for {len(texts)} messages...")
    # Generate embeddings (this will return a numpy array of shape (N, 384))
    message_embeddings = model.encode(texts)
    
    # We will test the search with a specific query
    query = "billing complaints or SLA breach"
    print(f"\n--- Testing Semantic Search ---")
    print(f"Query: '{query}'\n")
    
    # Embed the query
    query_embedding = model.encode([query])
    
    # Calculate cosine similarity between query and all messages
    # cosine_similarity returns a matrix of shape (1, N)
    similarities = cosine_similarity(query_embedding, message_embeddings)[0]
    
    # Get top 5 most similar messages by sorting indices in descending order
    top_k = 5
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    for rank, idx in enumerate(top_indices, 1):
        msg = messages[idx]
        score = similarities[idx]
        print(f"Rank {rank} (Score: {score:.4f}) | Account: {msg['account_id']} | Date: {msg['date'][:10]}")
        print(f"Text: {msg['text']}\n")
        
if __name__ == "__main__":
    main()
