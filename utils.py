import numpy as np
import openai

openai.api_key = "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"

def get_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def cosine_similarity(vec_a, vec_b):
    if not isinstance(vec_a, np.ndarray):
        vec_a = np.array(vec_a)
    if not isinstance(vec_b, np.ndarray):
        vec_b = np.array(vec_b)
    
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)


def semantic_search(documents, embeddings, query, top_k=5):
    query_embedding = get_embedding(query)
    if not query_embedding:
        return None, None, None
    
    # Calculate similarity scores
    similarities = []
    for i, embedding in enumerate(embeddings):
        if embedding is None:
            similarity = 0
        else:
            similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((i, similarity))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top k results
    top_indices = [item[0] for item in similarities[:top_k]]
    top_scores = [item[1] for item in similarities[:top_k]]
    top_documents = [documents[i] for i in top_indices]
    
    return top_documents, top_scores, query_embedding

def format_context(documents, scores):
    context = ""
    for i, (document, score) in enumerate(zip(documents, scores)):
        context += f"\n--- Document Chunk {i+1} (Similarity: {score:.4f}) ---\n"
        context += document
        context += "\n"
    return context

def generate_response(query, context):
    try:
        prompt = f"""
        You are an assistant that answers questions based on the given context.
        
        Context:
        {context}
        
        Question: 
        {query}
        """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        print(response)
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"Sorry, I couldn't generate a response. Error: {e}"
