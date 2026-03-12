from openai import OpenAI
import json
import numpy as np
from dotenv import load_dotenv
import os
import tiktoken
from tqdm import tqdm

# Load OPENAI_KEY
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tokenizer = tiktoken.get_encoding("cl100k_base")

def chunk_text(text, max_tokens=800):
    """Splits Japanese text into chunks of max_tokens."""
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i + max_tokens])

all_embeddings = []
metadata_store = []

# Load file to count total lines for tqdm
file_path = 'data_jp/jp_data.jsonl'
with open(file_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

with open(file_path, 'r', encoding='utf-8') as f:
    # Adding tqdm progress bar
    for line in tqdm(f, total=total_lines, desc="Processing Japanese Data"):
        item = json.loads(line)
        
        # Combine fields for context
        full_text = (
            f"Content: {item['content']}"
        )
        
        # Chunking to avoid the 400 Context Length error
        for chunk in chunk_text(full_text):
            try:
                response = client.embeddings.create(
                    input=chunk,
                    model="text-embedding-3-large"
                )
                all_embeddings.append(response.data[0].embedding)
                
                # Each chunk points back to the same URL/Metadata
                metadata_store.append({
                    "url": item.get("url", "No URL"),
                    "chunk_text": chunk # Store the specific chunk for the LLM
                })
            except Exception as e:
                print(f"\nError processing chunk: {e}")

# Save results
np.save('jp_vectors_RAG.npy', np.array(all_embeddings))
with open('jp_metadata_RAG.json', 'w', encoding='utf-8') as f:
    json.dump(metadata_store, f, ensure_ascii=False)

print("\nSuccess! Embeddings saved and ready for search.")