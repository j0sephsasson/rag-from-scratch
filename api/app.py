from flask import Flask, request, jsonify
import numpy as np
import json
from typing import Dict, List, Any
from llama_cpp import Llama
import torch
import logging
from transformers import AutoModel, AutoTokenizer

app = Flask(__name__)

# Set the logger level for Flask's logger
app.logger.setLevel(logging.INFO)

def compute_embeddings(text):
    tokenizer = AutoTokenizer.from_pretrained("/app/model/tokenizer") 
    model = AutoModel.from_pretrained("/app/model/embedding")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True) 
    
    # Generate the embeddings 
    with torch.no_grad():    
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()

    return embeddings.tolist()

def compute_matches(vector_store, query_str, top_k):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.
    """
    # Get the embedding for the query string
    query_str_embedding = np.array(compute_embeddings(query_str))
    scores = {}

    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    for doc_id, chunks in vector_store.items():
        for chunk_id, chunk_embedding in chunks.items():
            chunk_embedding_array = np.array(chunk_embedding)
            # Normalize embeddings to unit vectors for cosine similarity calculation
            norm_query = np.linalg.norm(query_str_embedding)
            norm_chunk = np.linalg.norm(chunk_embedding_array)
            if norm_query == 0 or norm_chunk == 0:
                # Avoid division by zero
                score = 0
            else:
                score = np.dot(chunk_embedding_array, query_str_embedding) / (norm_query * norm_chunk)

            # Store the score along with a reference to both the document and the chunk
            scores[(doc_id, chunk_id)] = score

    # Sort scores and return the top_k results
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_results = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]

    return top_results

def open_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def retrieve_docs(doc_store, matches):
    top_match = matches[0]
    doc_id = top_match[0]
    chunk_id = top_match[1]
    docs = doc_store[doc_id][chunk_id]
    return docs

def construct_prompt(system_prompt, retrieved_docs, user_query):
    prompt = f"""{system_prompt}

    Here is the retrieved context:
    {retrieved_docs}

    Here is the users query:
    {user_query}
    """
    return prompt

@app.route('/rag_endpoint', methods=['GET', 'POST'])
def main():
    app.logger.info('Processing HTTP request')

    # Process the request
    query_str = request.args.get('query') or (request.get_json() or {}).get('query')
    if not query_str:
        return jsonify({"error":"missing required parameter 'query'"})

    vec_store = open_json('/app/vector_store.json')
    doc_store = open_json('/app/doc_store.json')

    matches = compute_matches(vector_store=vec_store, query_str=query_str, top_k=3)
    retrieved_docs = retrieve_docs(doc_store, matches)

    system_prompt = """
    You are an intelligent search engine. You will be provided with some retrieved context, as well as the users query.

    Your job is to understand the request, and answer based on the retrieved context.
    """

    base_prompt = construct_prompt(system_prompt=system_prompt, retrieved_docs=retrieved_docs, user_query=query_str)

    app.logger.info(f'constructed prompt: {base_prompt}')

    # Formatting the base prompt
    formatted_prompt = f"Q: {base_prompt} A: "
    
    llm = Llama(model_path="/app/mistral-7b-instruct-v0.2.Q3_K_L.gguf")
    response = llm(formatted_prompt, max_tokens=800, stop=["Q:", "\n"], echo=False, stream=False)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)