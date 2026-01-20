from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

DB_PATH = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

llm = ChatOllama(
    model="phi3:mini",
    temperature=0.2
)


SYSTEM_PROMPT = """
You are an educational assistant. Answer strictly from the provided context.
If the answer is not present in the context, say:
"I could not find this in the provided documents."

End every answer with a clear conclusion.
"""

while True:
    query = input("\nAsk: ")
    if query.lower() in ["exit", "quit"]:
        break

    docs = retriever.invoke(query)
    context = "\n\n".join(d.page_content[:800] for d in docs)
    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)
    print("\nAnswer:\n", response.content)

