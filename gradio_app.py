import gradio as gr
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import os
from dotenv import load_dotenv

load_dotenv()
# ---------- LLM (Groq) ----------
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL")
)


# ---------- Vector DB ----------
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

chroma = chromadb.PersistentClient(path="db")

collection = chroma.get_or_create_collection(
    name="threat_docs",
    embedding_function=embedding_function
)

print("Docs in DB:", collection.count())

def answer(query):
    try:
        print("🔎 Query:", query)

        results = collection.query(
            query_texts=[query],
            n_results=3,
            include=["documents"]
        )

        docs = results["documents"][0]

        # Always use retrieved docs
        relevant_chunks = docs

        context = "\n\n".join(relevant_chunks)

        prompt = f"""
Use ONLY the information below.
If missing, say 'Not found in knowledge base.'

Context:
{context}

Question: {query}

Explain clearly:
1. What it means
2. Why it matters
3. Recommended actions
"""

        print("✉️ Sending to Groq...")

        res = client.chat.completions.create(
        model="openai/gpt-oss-120b", # Using a Groq-supported model
        messages=[{"role": "user", "content": prompt}]
    	)

        print("✅ LLM responded")
        return res.choices[0].message.content

    except Exception as e:
        print("❌ ERROR:", e)
        return f"❌ Error: {e}"



ui = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Paste alert / question", lines=4, placeholder="Multiple failed logins from same IP"),
    outputs=gr.Markdown(label="Result"),
    title="🛡️ RAG Cyber Threat Explainer",
    description="Ask about security alerts — RAG explains using your threat knowledge base."
)

ui.launch()

