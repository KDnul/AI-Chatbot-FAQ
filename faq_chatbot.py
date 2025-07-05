import json
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
import requests

# FAQ Database
# FAQ Questions
faq_questions = [
    "What are your business hours?",
    "How do I reset my password?",
    "Where are you located?",
    "How can I contact support?",
    "What is your refund policy?"
]

#FAQ Answers
faq_answers = [
    "Our business hours are Monday through Friday, 8:30am to 5pm.",
    "To reset your password, go to the login page and click 'Forgot Password'.",
    "We are located at 1234 Main Street, Anytown, USA.",
    "You can contact support via email at support@example.com.",
    "We offer a 30-day money-back gaurantee on all purchases."
]

# Load embedding model and build index\
model = SentenceTransformer('all-MiniLM-L6-v2')
faq_embeddings = model.encode(faq_questions)
index = faiss.IndexFlatL2(faq_embeddings.shape[1])
index.add(np.array(faq_embeddings))

# Search FAQ and get best match
def find_best_faq_answer(user_question):
    user_embedding = model.encode([user_question])
    D, I = index.search(np.array(user_embedding), k=1)
    best_index = I[0][0]
    similarity_score = D[0][0]
    return faq_questions[best_index], faq_answers[best_index], similarity_score

# Ask Ollama to rewrite answer naturally
def ask_ollama(question, context):
    prompt = f"Context:\n{context}\n\nUser Question:\n{question}"
    res = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3", "prompt": prompt},
        stream=True  # <-- Enable streaming
    )

    print("🤖 Bot: ", end ="", flush = True)
    reply = ""
    for line in res.iter_lines():
        if line:
            try:
                part = json.loads(line.decode("utf-8"))
                chunk = part.get("repsonce", "")
                reply += part.get("response", "")
                print(chunk, end="", flush=True)
            except json.JSONDecodeError:
                continue  # skip malformed line
    print("\n")
    return reply.strip() or "Sorry, I couldn't generate a response."

# Chat loop
def chat():
    print("🤖 FAQ Chatbot (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        matched_question, matched_aswer, score = find_best_faq_answer(user_input)
        print(f"\n[Matched FAQ: {matched_question} (Similarity: {score:.2f})]")

        if score > 1.0:
            print("🤖 Bot: Hmm, I'm not sure. Let me generate a helpful response...")
            reply = ask_ollama(user_input, "")
        else:
            reply = ask_ollama(user_input, matched_aswer)

        print(f"🤖 Bot: {reply}\n")

if __name__ == "__main__":
    chat()

