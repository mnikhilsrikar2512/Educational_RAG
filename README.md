# Educational RAG (Offline, Local LLM)

A Retrieval-Augmented Generation system that answers questions from educational PDFs
(AI, Algorithms, CS, Python).

Runs 100% locally:
- FAISS for vector search
- HuggingFace embeddings
- Ollama for LLM (no API keys, no cloud)

## Setup

```bash
git clone https://github.com/your-username/educational-rag.git
cd educational-rag
python -m venv rag
source rag/bin/activate
pip install -r requirements.txt
