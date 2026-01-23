import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

DB_PATH = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

pipe = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    max_new_tokens=300,
    temperature=0.2,
)

llm = HuggingFacePipeline(pipeline=pipe)

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
