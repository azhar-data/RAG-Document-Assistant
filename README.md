# 📚 RAG Document Assistant

A document question-answering app built with **Python, LangChain, ChromaDB, Hugging Face embeddings, Streamlit, and OpenRouter**.

Upload PDF or TXT files, create a semantic-search index, and ask questions. The app retrieves relevant passages and shows source filenames and page numbers when available.

## Features
- Upload multiple PDF and TXT files
- Split documents into overlapping text chunks
- Local sentence-transformer embeddings (`all-MiniLM-L6-v2`)
- In-memory Chroma vector store
- OpenRouter chat model integration
- Answers grounded in retrieved context
- Source snippets with filenames and PDF page numbers
- Streamlit chat interface
- API keys loaded from `.env`

## Architecture
1. **Load** — extract text from PDFs/TXT files.
2. **Chunk** — split text into overlapping pieces.
3. **Embed** — convert chunks into vectors locally.
4. **Index** — store vectors in an in-memory Chroma collection.
5. **Retrieve** — find the four most similar chunks for a question.
6. **Generate** — ask the selected OpenRouter model to answer using retrieved context.
7. **Show sources** — display retrieved source snippets.

## Requirements
- Python 3.10+
- An OpenRouter API key and a model available to your account
- Internet connection for package/model downloads and API requests

## Run on Windows (VS Code)
```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API key:
```env
OPENROUTER_API_KEY=your_real_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
```
Choose a model currently available to your OpenRouter account. Never commit your real API key.

Run the app:
```powershell
streamlit run app.py
```
Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Notes
- The first run may download the embedding model.
- The Chroma index is kept in memory for the current app session and is rebuilt when you click **Build / Refresh document index**.
- Scanned PDFs containing images rather than selectable text need OCR; OCR is not included.
- Retrieved passages can still be incomplete or misleading. Check source snippets before relying on an answer.
- Do not upload confidential documents to a public repository. Never commit `.env` or API keys.

## Project Structure
```text
RAG-Document-Assistant/
├── app.py
├── rag_app/
│   ├── __init__.py
│   └── pipeline.py
├── data/
│   └── .gitkeep
├── tests/
│   └── test_project_structure.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Author
**Muhammad Azhar**
- GitHub: https://github.com/azhar-data
- LinkedIn: https://www.linkedin.com/in/azhar-ai-engineer
