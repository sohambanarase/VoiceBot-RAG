import os
import json
import faiss
import numpy as np
import pandas as pd
from openai import AzureOpenAI

DATA_FOLDER = "data"
EMBEDDINGS_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBEDDINGS_FOLDER, "faiss_index.bin")
METADATA_FILE = os.path.join(EMBEDDINGS_FOLDER, "faiss_index.json")

# Initialize the Azure OpenAI client.
client = AzureOpenAI(
    api_key="xxxx",
    api_version="xxx",
    azure_endpoint="xxxx"
)

MODEL_NAME = "text-embedding-3-large"

def chunk_text(text, chunk_size=100, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        start += (chunk_size - overlap)
    return chunks

def load_csv_any_delimiter(filepath):
    try:
        df = pd.read_csv(filepath, sep=';', dtype=str, on_bad_lines='skip')
        return df
    except:
        pass
    try:
        df = pd.read_csv(filepath, sep=',', dtype=str, on_bad_lines='skip')
        return df
    except:
        pass
    try:
        import csv
        with open(filepath, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            dialect = csv.Sniffer().sniff(sample)
            f.seek(0)
            df = pd.read_csv(f, sep=dialect.delimiter, dtype=str, on_bad_lines='skip')
        return df
    except Exception as e:
        print(f"Could not parse CSV {filepath}: {e}")
        return pd.DataFrame()

def df_to_row_chunks(df, fname, max_chunk_size=100, overlap=50):
    row_chunks = []
    for row_idx, row in df.iterrows():
        content_dict = {
            col_name: "" if pd.isna(col_value) else str(col_value)
            for col_name, col_value in row.items()
        }
        # Create a unified text by joining each field with a semicolon and space.
        row_text_parts = [f"{col_name}: {content_dict[col_name]}" for col_name in content_dict]
        unified_text = " ; ".join(row_text_parts)
        # Split the unified text into overlapping subchunks
        subchunks = chunk_text(unified_text, chunk_size=max_chunk_size, overlap=overlap)
        
        part_num = 1
        for sc in subchunks:
            if sc.strip():
                meta = {
                    "filename": fname,
                    "row_index": row_idx,
                    "chunk_part": part_num,
                    "content": sc
                }
                row_chunks.append((sc, meta))
                part_num += 1
    return row_chunks

def extract_text_from_file(filepath, ext):
    # For text-based files like .txt and .log
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def embed_text(text_chunks):
    embeddings = []
    for text_chunk in text_chunks:
        response = client.embeddings.create(
            input=text_chunk,
            model=MODEL_NAME 
        )
        # Extract the embedding
        embedding = np.array(response.model_dump()["data"][0]["embedding"], dtype=np.float32)
        # Normalize the embedding to unit length (for cosine similarity)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        embeddings.append(embedding)
    return np.array(embeddings)

def main():
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)

    all_text_chunks = []
    all_metadata = []
    chunk_id = 0

    for fname in os.listdir(DATA_FOLDER):
        filepath = os.path.join(DATA_FOLDER, fname)
        ext = os.path.splitext(fname)[1].lower()

        row_chunks = []
        if ext == ".csv":
            df = load_csv_any_delimiter(filepath)
            if df.empty:
                print(f"Skipping empty/unreadable CSV: {fname}")
                continue
            row_chunks = df_to_row_chunks(df, fname)
        elif ext == ".xlsx":
            try:
                df = pd.read_excel(filepath, dtype=str)
                row_chunks = df_to_row_chunks(df, fname)
            except Exception as e:
                print(f"Error reading XLSX '{fname}': {e}")
                continue
        elif ext in [".txt", ".log"]:
            file_text = extract_text_from_file(filepath, ext)
            if file_text is None or not file_text.strip():
                print(f"Skipping empty or unreadable file: {fname}")
                continue
            # Break the file content into overlapping chunks using chunk_text
            text_chunks = chunk_text(file_text, chunk_size=100, overlap=50)
            for i, chunk in enumerate(text_chunks):
                if chunk.strip():
                    meta = {
                        "filename": fname,
                        "chunk_index": i,
                        "content": chunk
                    }
                    row_chunks.append((chunk, meta))
        else:
            print(f"Skipping unsupported file type: {fname}")
            continue

        for text_chunk, meta in row_chunks:
            all_text_chunks.append(text_chunk)
            meta["id"] = chunk_id
            all_metadata.append(meta)
            chunk_id += 1

    print(f"Total chunks to embed: {len(all_text_chunks)}")

    # Embed and build FAISS index using cosine similarity (normalized embeddings)
    embeddings = embed_text(all_text_chunks)
    dim = embeddings.shape[1]
    # Use IndexFlatIP for inner product search (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Write FAISS index and metadata to disk
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print("FAISS index and metadata saved. Done!")

if __name__ == "__main__":
    main()
