# 🧠 Simple Retrieval-Augmented Generation (RAG) with Ollama

This is a minimal example of how to build a **Retrieval-Augmented Generation (RAG)** system using [Ollama](https://ollama.com/). It loads a small dataset (cat facts 🐱), generates vector embeddings, retrieves the most relevant entries for a user query, and uses a language model to generate a response based only on the retrieved context.

> ✅ Inspired by the blog post: [Make Your Own RAG](https://huggingface.co/blog/ngxson/make-your-own-rag) by Son Nguyen (ngxson).

---

## 🚀 Features

- Uses **Ollama** to generate embeddings and chat responses.
- Employs **cosine similarity** for retrieval of top-k relevant chunks.
- Constructs a system prompt that limits LLM to only retrieved context.
- Streams chatbot responses in real-time.
- Fully self-contained – no additional dependencies beyond Ollama and a text file.

---

## 📦 Requirements

- [Python 3.8+](https://www.python.org/)
- [Ollama](https://ollama.com/) installed and running on your machine
- A working `cat-facts.txt` dataset (one fact per line)

---

## 🔧 How to Run

1. **Install Ollama and pull the required models:**

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

2. **Run the script
```bash
python simple_rag.py
```

3. Ask question when prompted




