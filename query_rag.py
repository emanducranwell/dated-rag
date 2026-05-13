import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("british_museum_objects")

question = "How did you get into the museum?"

query_embedding = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
    print("\n--- RESULT ---")
    print(metadata)
    print(doc[:800])