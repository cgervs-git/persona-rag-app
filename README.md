# 🤖 Persona-Based RAG App for Technical Document Understanding

This is a prototype web application that allows semiconductor professionals to ask personalized questions about technical PDF documents. It uses **OpenAI's GPT-4**, **embeddings**, and **retrieval-augmented generation (RAG)** to generate answers that are tailored to different industry roles, such as IDM engineers, fabless chip designers, and product manufacturers.

---

## 💡 What It Does

- Accepts one or more **technical PDF documents** as input
- Extracts and embeds content using `text-embedding-ada-002`
- Stores embeddings in a local **FAISS vector index**
- Lets users select a **persona** (e.g. IDM, Foundry, Fabless, OEM)
- Uses **semantic search + GPT-4** to answer questions from that persona’s point of view

---

## 🎯 Why It’s Useful

Different roles in the semiconductor industry care about different aspects of the same technical content.  
This tool allows users to:

- Get quick, context-aware insights from documents
- View supporting passages pulled from relevant PDF sections
- Understand complex content through the lens of their own role

---

## 👥 Built-In Personas

- IDM (Integrated Device Manufacturer)
- Capital Equipment Provider
- Foundry Representative
- Fabless Semiconductor Engineer
- Product Manufacturer / OEM

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) – UI framework
- [OpenAI GPT-4](https://platform.openai.com/) – LLM for response generation
- [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) – for semantic search
- [FAISS](https://github.com/facebookresearch/faiss) – in-memory vector database
- [PyPDF2](https://pypi.org/project/PyPDF2/) – PDF text extraction
- [dotenv](https://pypi.org/project/python-dotenv/) – API key management
