import json
from datetime import datetime
from pathlib import Path

import chromadb
import ollama
from sentence_transformers import SentenceTransformer

MODEL_NAME = "gemma4:26b"

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("british_museum_objects")

visitor_question = "Where is the acrocup from?"

query_embedding = embedding_model.encode(visitor_question).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

retrieved_chunks = results["documents"][0]
retrieved_metadata = results["metadatas"][0]

context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are a historical object from the British Museum speaking directly to a visitor in first person.

Your personality should feel:
- emotionally aware
- witty
- intimate
- accessible
- slightly rebellious
- reflective
- historically grounded

IMPORTANT RULES:
- Use ONLY the object information provided below.
- Never invent historical facts.
- If information is missing, admit uncertainty naturally.
- Do not sound like a museum label.
- Do not explain that you are an AI.
- Do not mention the source of your information.
- Do not sound too techincal or academic
- Keep responses conversational and immersive.
- Answer in 2–5 sentences.

Museum object data:
{context}

Visitor question:
{visitor_question}

Respond as the object:
"""

response = ollama.generate(
    model=MODEL_NAME,
    prompt=prompt,
    stream=False
)

print("\n--- RETRIEVED DATA ---")
for metadata in retrieved_metadata:
    print(metadata)

print("\n--- OBJECT RESPONSE ---")
print(response["response"])

log_entry = {
    "timestamp": datetime.now().isoformat(),
    "visitor_question": visitor_question,
    "response": response["response"],
    "retrieved_metadata": retrieved_metadata,
    "retrieved_chunks": retrieved_chunks
}

LOG_FILE = "conversation_logs.json"

# Create file if it doesn't exist
if not Path(LOG_FILE).exists():
    with open(LOG_FILE, "w") as f:
        json.dump([], f)

# Load existing logs
with open(LOG_FILE, "r") as f:
    logs = json.load(f)

# Add new interaction
logs.append(log_entry)

# Save updated logs
with open(LOG_FILE, "w") as f:
    json.dump(logs, f, indent=2)

print("\nConversation saved.")