# persona_rag_app.py

import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
import openai
import faiss
import numpy as np
from typing import List

# Load API key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Constants
EMBED_MODEL = "text-embedding-ada-002"
CHUNK_SIZE = 500
INDEX_DIM = 1536  # dim for ada-002 embeddings

# Memory store
doc_chunks = []  # Holds (chunk_text, source, persona_tag)
index = faiss.IndexFlatL2(INDEX_DIM)

# Load personas
@st.cache_data
def load_personas(file_path="personas.txt"):
    personas = {}
    current_key = None
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                current_key = line[1:].strip()
            elif current_key and line:
                personas[current_key] = line
                current_key = None
    return personas

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

# Chunk text
def chunk_text(text: str, chunk_size=CHUNK_SIZE) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Embed text using OpenAI
@st.cache_data(show_spinner=False)
def embed_texts(texts: List[str]) -> np.ndarray:
    embeddings = []
    for i in range(0, len(texts), 20):
        batch = texts[i:i+20]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeds = [e.embedding for e in response.data]
        embeddings.extend(batch_embeds)
    return np.array(embeddings).astype("float32")

# Search top-k chunks
def search_chunks(query, top_k=5):
    query_embed = embed_texts([query])[0].reshape(1, -1)
    distances, indices = index.search(query_embed, top_k)
    return [doc_chunks[i] for i in indices[0]]

# Ask GPT
def ask_persona_question(query, persona_desc, context_chunks):
    context = "\n---\n".join([chunk[0] for chunk in context_chunks])
    prompt = f"""
You are responding as a {persona_desc}.
Use the following technical context to answer the question.

Context:
{context}

Question: {query}

Respond in a professional tone tailored to the needs and pain points of this role.
"""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.set_page_config(page_title="Persona-Based RAG Assistant", layout="wide")
st.title("🔎 Persona-Based Technical Assistant (RAG Prototype)")

# Upload and embed documents
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Choose one or more technical PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.success(f"Uploaded: {uploaded_file.name}")
        raw_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(raw_text)
        embeds = embed_texts(chunks)

        # Store chunks and index
        for i, chunk in enumerate(chunks):
            doc_chunks.append((chunk, uploaded_file.name, None))
        index.add(embeds)

    st.info(f"Indexed a total of {len(doc_chunks)} chunks from {len(uploaded_files)} documents.")

# Query section
personas = load_personas()
persona_label = st.selectbox("Select Your Persona", list(personas.keys()))
persona_desc = personas[persona_label]

query = st.text_input("Ask a question related to the uploaded documents:")
if st.button("Ask") and query:
    top_chunks = search_chunks(query)
    response = ask_persona_question(query, persona_desc, top_chunks)
    st.subheader(f"Answer for {persona_label} Persona")
    st.write(response)
    with st.expander("🔍 View Supporting Context"):
        for chunk, source, _ in top_chunks:
            st.markdown(f"**From {source}**\n> {chunk[:300]}...")
