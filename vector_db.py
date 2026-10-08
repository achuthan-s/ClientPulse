import json
import os
# pyrefly: ignore [missing-import]
import chromadb
from chromadb.utils import embedding_functions

def load_data(filepath="data/extracted_messages.json"):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run previous phases.")
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    print("Loading extracted messages...")
    messages = load_data()
    
    # Initialize ChromaDB. We'll use a persistent client so the database is saved to disk.
    db_path = "./chroma_db"
    print(f"Initializing ChromaDB at {db_path}...")
    client = chromadb.PersistentClient(path=db_path)
    
    # ChromaDB uses sentence-transformers/all-MiniLM-L6-v2 by default.
    # We explicitly define the embedding function for clarity.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Get or create a collection for our client messages
    collection_name = "client_messages"
    
    # If we run this script multiple times, we want a fresh start
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass # Collection didn't exist yet
        
    collection = client.create_collection(
        name=collection_name, 
        embedding_function=sentence_transformer_ef
    )
    
    # Prepare data for insertion
    ids = []
    documents = []
    metadatas = []
    
    for msg in messages:
        ids.append(msg["message_id"])
        documents.append(msg["text"])
        
        # ChromaDB metadata can only hold strings, ints, floats, or bools.
        # We join the topics list into a single comma-separated string.
        metadatas.append({
            "account_id": msg["account_id"],
            "date": msg["date"],
            "channel": msg["channel"],
            "sentiment": float(msg["sentiment"]),
            "topics": ",".join(msg.get("topics", [])),
            "scope_dispute_signal": bool(msg.get("scope_dispute_signal", False))
        })
        
    print(f"Adding {len(documents)} messages to the vector database...")
    # Add documents to the collection. Chroma automatically embeds the 'documents' list using the embedding function.
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print("Database population complete.\n")
    
    # --- Test 1: Simple Semantic Search ---
    query_1 = "missing endpoints"
    print(f"Test 1: Search for '{query_1}' across ALL accounts")
    results_1 = collection.query(
        query_texts=[query_1],
        n_results=2
    )
    
    for idx, doc in enumerate(results_1['documents'][0]):
        account = results_1['metadatas'][0][idx]['account_id']
        print(f" - [{account}]: {doc}")
        
    print("\n--------------------------------------------------\n")
        
    # --- Test 2: Semantic Search WITH Metadata Filtering ---
    # We know TechCorp complained about endpoints, but what if we want to search
    # only within the healthy account (InnoVate) to see if they ever had endpoint issues?
    
    query_2 = "missing endpoints"
    filter_account = "acct_innovate"
    print(f"Test 2: Search for '{query_2}' ONLY within {filter_account}")
    
    results_2 = collection.query(
        query_texts=[query_2],
        n_results=2,
        where={"account_id": filter_account} # THIS IS THE MAGIC OF METADATA FILTERING
    )
    
    for idx, doc in enumerate(results_2['documents'][0]):
        account = results_2['metadatas'][0][idx]['account_id']
        print(f" - [{account}]: {doc}")

if __name__ == "__main__":
    main()
