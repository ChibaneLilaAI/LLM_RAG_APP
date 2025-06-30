from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# FastAPI app
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load and prepare data
BIO_PATH = "biographie_lila.txt"
print("📄 Chargement de la biographie...")
with open(BIO_PATH, "r", encoding="utf-8") as f:
    text = f.read()

print("🔪 Découpage en chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0, separators=["\n\n"])
chunks = splitter.split_text(text)

print("🧠 Embeddings...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks, convert_to_numpy=True)

print("🔍 Index FAISS...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

def answer_question(question, top_k=4):
    query_vec = embedding_model.encode([question], convert_to_numpy=True)
    scores, indices = index.search(query_vec, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join(retrieved_chunks)

    prompt = f"Voici un contexte :\n{context}\n\nQuestion : {question}\nRéponse :"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Tu es un expert qui répond précisément aux questions en te basant uniquement sur le contexte fourni."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

@app.get("/", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
def handle_question(request: Request, question: str = Form(...)):
    answer = answer_question(question)
    return templates.TemplateResponse("form.html", {
        "request": request,
        "question": question,
        "answer": answer
    })
