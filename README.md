# SourceCodeAnalyzer

SourceCodeAnalyzer is an AI-powered GitHub repository analysis tool that uses **Retrieval-Augmented Generation (RAG)** to understand and answer questions about source code. Users can provide a public GitHub repository URL, and the application automatically clones the repository, processes the Python source files, creates embeddings, and stores them in **ChromaDB**. When a user asks a question, the system retrieves the most relevant code chunks and provides a context-aware answer using an OpenAI language model.

## Features

* Analyze public GitHub repositories
* Automatically clone and process source code
* Code chunking and embedding generation
* ChromaDB vector storage
* MMR-based relevant code retrieval
* Context-aware conversational Q&A
* Support for switching between repositories
* Simple web-based chat interface

## Tech Stack

**Python | Flask | LangChain | ChromaDB | OpenAI | GitPython | HTML | CSS | JavaScript**

## Project Structure

```text
SourceCodeAnalyzer/
├── src/
│   └── helper.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── app.py
├── store_index.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Setup

```bash
conda create -n codeanalyse python=3.11
conda activate codeanalyse
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

Run the application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:8080
```

## Example Questions

* Explain the project structure.
* Where is authentication implemented?
* How does the API work?
* Explain this function.
* Which files handle the database?


