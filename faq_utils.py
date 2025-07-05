import json
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
import requests

# Sample FAQs
faqs = [
    {"question": "What are your business hours?", "answer": "We are open from 8:30am to 5pm PST, Monday to Friday."},
    {"question": "Where are you located?", "answer": "We are located at 1234 main Street, Anytown, USA."},
    {"question": "How can I contact support?", "answer": "You can email us at support@example.com."}
]

# Embed Setup
model = SentenceTransformer('all-MiniLM-L6-v2')
faq_questions = [faq["question"] for faq in faqs]
faq_embeddings = model.encode(faq_questions)
index = faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(np.array(faq_embeddings))

def find_best_faq_answer(user_input):
    user_embedding = model.encode([user_input])
    distance, index_result = index.search(np.array(user_embedding), 1)
    match = faqs[index_result[0][0]]
    return match["question"], match["answer"], distance[0][0]

def ask_ollama(user_question, context):
    prompt = f"Context:\n{context}\n\nUser Question:\n{user_question}"
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt},
        stream=True
    )

    reply = ""
    for line in res.iter_lines():
        if line:
            try:
                part = json.loads(line.decode("utf-8"))
                reply += part.get("response", "")
            except json.JSONDecodeError:
                continue
    return reply.strip() or "Sorry, I couldn't generate a response."