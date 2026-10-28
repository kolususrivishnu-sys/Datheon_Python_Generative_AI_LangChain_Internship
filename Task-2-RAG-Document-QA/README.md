
# Document Q&A Assistant with RAG & Vector Embeddings

A Python application that answers questions about custom PDF and TXT
documents using Retrieval-Augmented Generation (RAG).

The application loads documents, splits their text into chunks,
creates vector embeddings, stores them in ChromaDB, and retrieves
relevant context to answer questions using Google Gemini.

## Features

- Load PDF and TXT documents
- Split long documents into overlapping text chunks
- Generate vector embeddings using Google Gemini
- Store embeddings in a local ChromaDB database
- Retrieve relevant context for user questions
- Generate context-grounded answers using Gemini
- Display source document names and PDF page numbers
- Interactive command-line question-answering interface

## Technology Stack

- Python 3.11+
- LangChain
- Google Gemini API
- ChromaDB
- Pydantic-compatible LangChain components
- PyPDF
- python-dotenv

## Project Structure

```text
Task-2-RAG-Document-QA/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── data/
│   └── sample_document.txt
└── chroma_db/              # Created locally at runtime
```

## Installation

Open Terminal and navigate to this project folder:

```bash
cd Task-2-RAG-Document-QA
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## API Configuration

Create a local `.env` file by copying `.env.example`:

```bash
cp .env.example .env
```

Add your Google Gemini API key to `.env`:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

Replace the placeholder with your own key.

Never commit `.env` or publish your real API key on GitHub.
API usage may incur charges according to your provider's plan.

## Run the Application

Place PDF or TXT files inside the `data/` folder.

Run:

```bash
python app.py --data-dir data --db-dir chroma_db
```

Wait for the documents to be loaded and indexed.
Enter a question when prompted.

Type `exit` to stop the application.

## Example Questions

Using the included fictional sample document, try:

- What is the Smart Library Assistant?
- Which technologies does the project use?
- How does the application process PDF files?
- How many relevant chunks are retrieved for each question?
- What should the assistant do if information is missing?

## How RAG Works

1. Document Loading: Read PDF and TXT files.
2. Text Splitting: Divide long documents into smaller chunks.
3. Embedding: Convert chunks into vectors using a Gemini embedding model.
4. Vector Storage: Store the vectors locally in ChromaDB.
5. Retrieval: Find relevant chunks for the question.
6. Generation: Ask Gemini to answer using the retrieved context.
7. Citations: Display the source document and available PDF page numbers.

## Limitations

- Requires a valid Gemini API key and internet connectivity.
- Answers depend on the quality of the source documents and retrieval.
- Scanned PDFs may require OCR before their text can be searched.
- Source references help trace answers but do not guarantee correctness.
- Live API integration should be tested before production use.

## Security

- Keep API credentials in a local `.env` file.
- Do not upload confidential documents or API keys.
- Use fictional documents for public demonstrations.

## Learning Outcomes

This project demonstrates document processing, recursive text splitting,
vector embeddings, semantic retrieval, vector databases, prompt design,
and source-aware question answering.
