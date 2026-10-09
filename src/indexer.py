"""
SmartMatch - Supplier Vector Indexer
Loads supplier data, computes vector embeddings using Gemini API (or local fallback),
stores vectors in persistent ChromaDB, and provides top-k retrieval functionality.
"""

import os
import time
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
EMBEDDING_MODEL = "text-embedding-004"
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "suppliers.csv"))
COLLECTION_NAME = "suppliers"
BATCH_SIZE = 20


def get_gemini_client():
    """Initializes Gemini API client if API key is configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        return None
    try:
        from google import genai
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"[!] Could not initialize Gemini SDK client: {e}")
        return None


def get_embeddings(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Computes vector embeddings for a list of texts using Gemini API with retry logic,
    falling back to local default embeddings if API key is missing or fails.
    """
    client = get_gemini_client()

    if client:
        try:
            from google.genai import types
            all_embeddings = []

            for i in range(0, len(texts), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                for attempt in range(3):
                    try:
                        response = client.models.embed_content(
                            model=EMBEDDING_MODEL,
                            contents=batch,
                            config=types.EmbedContentConfig(
                                task_type=task_type
                            )
                        )
                        batch_embeddings = [emb.values for emb in response.embeddings]
                        all_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        print(f"[!] Embedding API attempt {attempt + 1} failed: {e}")
                        time.sleep(2 ** attempt)

            if len(all_embeddings) == len(texts):
                return all_embeddings
            else:
                print("[!] Gemini API returned incomplete embeddings. Falling back to local embedder.")
        except Exception as e:
            print(f"[!] Gemini embedding failed: {e}. Falling back to local embedder.")

    # Fallback to local default embedding function (all-MiniLM-L6-v2)
    print("[-] Using local fallback embedding model (all-MiniLM-L6-v2)...")
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    return default_ef(texts)


def build_supplier_document(row: pd.Series) -> str:
    """Combines supplier metadata into a natural-language description for embedding."""
    return (
        f"Supplier Company: {row['company_name']} (ID: {row['supplier_id']}). "
        f"Location: {row['city']}, {row['state']}. "
        f"Industry Category: {row['category']}. "
        f"Offered Products: {row['products']}. "
        f"Specifications & Grades: {row['specifications']}. "
        f"Price Range: {row['price_range_inr']} INR. "
        f"Minimum Order Quantity (MOQ): {row['moq']}. "
        f"Years in Business: {row['years_in_business']}. "
        f"Verified Seller: {'Yes' if row['verified'] else 'No'}."
    )


def build_index():
    """Reads suppliers.csv, computes embeddings, and builds ChromaDB vector store."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Supplier data file not found at {DATA_PATH}. Run src/generate_data.py first.")

    print(f"[+] Loading supplier catalog from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)

    documents = []
    metadatas = []
    supplier_ids = []

    for _, row in df.iterrows():
        doc = build_supplier_document(row)
        documents.append(doc)
        supplier_ids.append(str(row['supplier_id']))
        metadatas.append({
            "supplier_id": str(row['supplier_id']),
            "company_name": str(row['company_name']),
            "city": str(row['city']),
            "state": str(row['state']),
            "category": str(row['category']),
            "verified": bool(row['verified'])
        })

    print(f"[+] Embedding {len(documents)} supplier documents...")
    embeddings = get_embeddings(documents, task_type="RETRIEVAL_DOCUMENT")

    print(f"[+] Initializing persistent ChromaDB store at {CHROMA_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Idempotent collection reset
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"[-] Existing collection '{COLLECTION_NAME}' deleted for clean rebuild.")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"[+] Adding {len(documents)} vectors to ChromaDB collection '{COLLECTION_NAME}'...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=supplier_ids
    )

    print(f"[OK] Indexing completed successfully. Total records indexed: {collection.count()}")


def search(query: str, k: int = 5) -> list[dict]:
    """
    Embeds the buyer search query and retrieves top-k matching suppliers from ChromaDB.

    Returns:
        List of dicts containing supplier_id, distance, metadata, and document text.
    """
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(f"ChromaDB store not found at {CHROMA_PATH}. Run build_index() first.")

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    query_embedding = get_embeddings([query], task_type="RETRIEVAL_QUERY")[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    formatted_results = []
    if results and results.get('ids') and len(results['ids']) > 0:
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        docs = results['documents'][0]

        for i in range(len(ids)):
            formatted_results.append({
                "supplier_id": ids[i],
                "distance": distances[i],
                "metadata": metadatas[i],
                "document": docs[i]
            })

    return formatted_results


if __name__ == "__main__":
    print("==================================================")
    print(" SmartMatch Vector Indexing & Retrieval Test")
    print("==================================================")

    # Build or rebuild index
    build_index()

    # Test Search Query
    test_query = "500 kg stainless steel sheet Delhi"
    print("\n" + "="*50)
    print(f" TESTING VECTOR RETRIEVAL FOR QUERY: '{test_query}'")
    print("="*50)

    top_matches = search(test_query, k=5)

    for rank, match in enumerate(top_matches, 1):
        meta = match['metadata']
        print(f"\nRank #{rank} | Supplier ID: {match['supplier_id']} | Cosine Distance: {match['distance']:.4f}")
        print(f"  Company : {meta.get('company_name')} ({meta.get('city')}, {meta.get('state')})")
        print(f"  Category: {meta.get('category')} | Verified: {meta.get('verified')}")
        print(f"  Doc Snippet: {match['document'][:120]}...")

    print("\n[OK] Search test finished.")
