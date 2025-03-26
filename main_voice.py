import os
import random
import torch
import json
import faiss
import numpy as np
import sounddevice as sd
from sentence_transformers import CrossEncoder
from openai import AzureOpenAI
from faster_whisper import WhisperModel
from bark import generate_audio, preload_models, SAMPLE_RATE


#  Configuration 
AZURE_OPENAI_API_KEY = 'xxx'
AZURE_OPENAI_ENDPOINT = 'xxxx'
AZURE_OPENAI_API_VERSION = 'xxxx'
DEPLOYMENT_NAME = 'xxxxx'

MODEL_NAME = "text-embedding-3-large"

EMBEDDINGS_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBEDDINGS_FOLDER, "faiss_index.bin")
METADATA_FILE = os.path.join(EMBEDDINGS_FOLDER, "faiss_index.json")

# Load FAISS Index and Metadata 
index = faiss.read_index(INDEX_FILE)
with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Initialize Azure OpenAI Client and Cross-Encoder 
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Dense Retrieval Functions
def embed_query(query):
    response = client.embeddings.create(
         input=query,
         model=MODEL_NAME
    )
    embedding = np.array(response.model_dump()["data"][0]["embedding"], dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return np.expand_dims(embedding, axis=0)

def retrieve_dense(query, candidate_k=200, score_threshold=0.3):
    query_emb = embed_query(query)
    scores, indices = index.search(query_emb, candidate_k)
    dense_candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if score_threshold is not None and score < score_threshold:
            continue
        dense_candidates.append((metadata[idx]["content"], metadata[idx], score))
    dense_candidates = sorted(dense_candidates, key=lambda x: x[2], reverse=True)
    print("\nDEBUG: Dense retrieval results:")
    for candidate in dense_candidates:
        print(candidate)
    return dense_candidates

def rerank_candidates(query, candidates, top_n=5):
    pairs = [(query, text) for text, meta, score in candidates]
    if not pairs:
        print("\nDEBUG: No candidates available for re-ranking!")
        return []
    rerank_scores = reranker.predict(pairs)
    scored_candidates = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
    top_candidates = [candidate for score, candidate in scored_candidates[:top_n]]
    print("\nDEBUG: Top candidates after re-ranking:")
    for candidate in top_candidates:
        print(candidate)
    return top_candidates

def build_prompt_with_context(user_query, retrieved_chunks):
    context_parts = []
    for chunk_text, meta, score in retrieved_chunks:
        index_info = meta.get('row_index', meta.get('chunk_index', 'N/A'))
        part_info = meta.get('chunk_part', '1')
        context_parts.append(
            f"From file '{meta['filename']}', index {index_info} (part {part_info}):\n{chunk_text}"
        )
    context_str = "\n\n".join(context_parts)
    system_instruction = (
        "You are a helpful assistant named bot who answers questions related to your company ''Your Company'' and answers within 200 characters. Respond naturally "
        "including human-like hesitations (e.g., 'umm', 'ahh') where appropriate, and give helpful suggestions if asked. "
        f"{context_str}\n\n"
        "If the context does not answer the question, say so."
    )
    conversation = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_query}
    ]
    print("\nDEBUG: Final context for GPT:")
    print({"role": "user", "content": user_query})
    return conversation

#  Audio Recording and Transcription 
def record_audio_array(duration=5, fs=16000):
    print("listening...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to complete
    return recording

def transcribe_audio_array(audio_array, fs=16000):
    # Convert int16 to float32
    audio_float = audio_array.astype(np.float32)
    # Normalize audio to [-1, 1]
    max_val = np.max(np.abs(audio_float))
    if max_val > 0:
        audio_float = audio_float / max_val
    # Flatten the array to 1D, as expected by faster-whisper
    audio_float = audio_float.flatten()
    
    print("Transcribing audio from memory...")
    segments, info = model.transcribe(audio_float, beam_size=5)
    transcription = " ".join([seg.text for seg in segments])
    print("Transcription:", transcription)
    return transcription

#  Initialize Models 
# Change device to "cuda" if you have a GPU available; here it is set to "cpu"
model = WhisperModel("small", device="cpu", compute_type="int8")
preload_models()  # Preload Bark TTS models

def speak_response(response_text):
    print("Converting text to speech...")
    # Use a fixed voice preset to ensure consistent output.
    fixed_voice = "v2/en_speaker_9"  # Choose your desired preset from Bark's voice prompt library.
    audio_array = generate_audio(response_text, history_prompt=fixed_voice)
    print("Playing audio response...")
    sd.play(audio_array, SAMPLE_RATE)
    sd.wait()

# --- Main Loop ---
def main():
    print("Voice Bot is running. Press Enter to record your query (or type 'exit' to quit).")
    while True:
        command = input("Press Enter to record or type 'exit': ")
        if command.strip().lower() == "exit":
            print("Exiting Voice Bot.")
            break

        # Record audio directly into memory (as a numpy array)
        audio_array = record_audio_array(duration=5, fs=16000)
        
        # Transcribe the audio from memory without saving to disk
        user_query = transcribe_audio_array(audio_array, fs=16000)
        if not user_query.strip():
            print("No transcription obtained. Please try again.")
            continue

        # Dense retrieval and GPT prompt construction
        dense_candidates = retrieve_dense(user_query, candidate_k=200, score_threshold=0.3)
        final_candidates = rerank_candidates(user_query, dense_candidates, top_n=5)
        conversation = build_prompt_with_context(user_query, final_candidates)

        # Get response from Azure OpenAI Chat model
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=conversation
        )
        answer = response.choices[0].message.content
        print("\nAssistant:", answer, "\n")
        speak_response(answer)

if __name__ == "__main__":
    main()
