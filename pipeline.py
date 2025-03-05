import os
import glob
import nltk
from nltk.tokenize import sent_tokenize
import chromadb
from utils import get_embedding
import json

def load_data(directory_path):
    file_paths = glob.glob(os.path.join(directory_path, "*.txt"))
    data = ""
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                data += content + "\n\n"
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return data

def chunk_data(data, max_words=500):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    sentences = sent_tokenize(data)

    chunks = []
    current_chunk = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) > max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            word_count = 0
        current_chunk.append(sentence)
        word_count += len(words)
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

if __name__ == "__main__":
    client = chromadb.PersistentClient(path="./db")
    collection = client.get_or_create_collection("knowledge_base")

    data = load_data('./data')

    chunks = chunk_data(data)

    data_chunked = []
    data_embed = []

    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding:
            data_embed.append(embedding)
            data_chunked.append(chunk)
        else:
            print(f"Failed to get embedding for chunk {i}")

    collection.add(
        documents=data_chunked,
        embeddings=data_embed,
        ids=[str(i) for i in range(len(data_chunked))]
    )