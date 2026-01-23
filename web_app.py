import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_PATH = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOllama(model="phi3:mini", temperature=0.2)

SYSTEM_PROMPT = """
You are an educational assistant. Answer strictly from the provided context.
If the answer is not present in the context, say:
"I could not find this in the provided documents."

End every answer with a clear conclusion.
"""

st.title("Educational RAG Assistant")

query = st.text_input("Ask a question")

if query:
    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content[:400] for d in docs[:2])

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    st.write(response.content)
