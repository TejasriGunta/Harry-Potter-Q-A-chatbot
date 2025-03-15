import PyPDF2 as pf
from PyPDF2 import PdfReader
reader=PdfReader("Harry Potter and the Prisoner of Azkaban.pdf")
text=""
for page in reader.pages:
    text+=page.extract_text()
chapters=text.split("Chapter")

def make_chunks(text):
    words=text.split()
    chunks=[]
    for i in range(0,len(words),120):
        chunk=" ".join(words[i:i+150])
        chunks.append(chunk)
    return chunks
        
chunks = make_chunks(text) 

from sentence_transformers import SentenceTransformer
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeds=model.encode(chunks) 

import faiss as fs
import numpy as np
embeds=np.array(embeds, dtype="float32")
dimension=embeds.shape[1]
index=fs.index_factory(dimension, "Flat", fs.METRIC_L2)
index.add(embeds)
metadata=[{'chunk_id':i,"text":chunks[i]} for i in range(len(chunks))]

from dotenv import load_dotenv
import os
import requests
import json

# Load the environment variables from the .env file
load_dotenv()
API_KEY= os.getenv("API_KEY")

# Define the URL with the API key loaded from the environment
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={API_KEY}"

# Set the headers
headers = {
    'Content-Type': 'application/json'
}

def get_response(prompt):
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        response_data = response.json()
        # Extract and return the text content from the response
        response=response_data['candidates'][0]['content']['parts'][0]['text']
        return(response)
        
    else:
        return(f"Error: {response.status_code} - {response.text}")
       

def get_query_and_response_gemini(query):
    query_embed=model.encode(query)
    query_embed=np.array(query_embed,dtype='float32').reshape(1,-1)
    distance, indices=index.search(query_embed, 3)
    results=[]
    context=" "
    for i in range(3):
        result_metadata=metadata[indices[0][i]]
        results.append(result_metadata)
        context+=(results[i]['text'])
        
    prompt=f"""You are an intelligent chatbot specialized in answering Harry Potter and the prisinor of azkaban-related questions.
    Your task:
    - If the retrieved context is relevant to the user's question, use it.
    - If the retrieved context is **irrelevant** or **empty**, ignore it and answer from your own knowledge.
    ### Context:
    # {context}
    # ### User Question:
    # {query}
    generate answer within 100-150 words,for introductory questions keep it short and creative, use your wizardy powers!"""
    response=get_response(prompt)
    return(response)
