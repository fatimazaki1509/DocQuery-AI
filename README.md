# DocQuery AI

### Document Intelligence & Retrieval-Augmented Question Answering

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/RAG-Pipeline-6C63FF?style=for-the-badge" alt="RAG">
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-00A67E?style=for-the-badge" alt="FAISS">
  <img src="https://img.shields.io/badge/Groq-LLM-F55036?style=for-the-badge" alt="Groq">
</p>

<p align="center">
  <a href="https://docquery-ai-fifnvr65d3nw4qqdvur2fd.streamlit.app/">
    <strong>Live Demo</strong>
  </a>
  &nbsp; • &nbsp;
  <a href="https://github.com/fatimazaki1509/DocQuery-AI">
    <strong>Source Code</strong>
  </a>
</p>

---

## Overview

**DocQuery AI** is a lightweight document question-answering application built using a Retrieval-Augmented Generation (RAG) architecture.

The application allows users to upload a PDF, ask multiple questions about the document, and receive answers based on the relevant content retrieved from that document.

Instead of passing the complete document to the language model for every question, DocQuery AI first converts the document into searchable vector representations and retrieves the most relevant sections for each query.

This creates a simple pipeline:

```text
PDF → Text Extraction → Chunking → Embeddings → FAISS
                                              ↓
Question → Embedding → Similarity Search → Relevant Context
                                              ↓
                                         Groq LLM
                                              ↓
                                            Answer
```

---

## Live Demo

**Try the application:**

### [DocQuery AI](https://docquery-ai-fifnvr65d3nw4qqdvur2fd.streamlit.app/)

Upload any text-based PDF and start asking questions about its content.

---

## Application Preview

> **Dashboard / Application Screenshot**

<img width="866" height="801" alt="image" src="https://github.com/user-attachments/assets/3ed049fd-02e2-4463-8694-763cc7a19154" />


---

# What It Does

DocQuery AI provides a simple interface for interacting with PDF documents using natural-language questions.

### Document Processing

The uploaded PDF goes through the following pipeline:

1. Text extraction using `pypdf`
2. Character-based chunking
3. Sentence Transformer embeddings
4. FAISS vector indexing

### Question Answering

When a question is submitted:

1. The question is converted into an embedding.
2. FAISS searches the document index.
3. The top five relevant chunks are retrieved.
4. The retrieved content is provided to the LLM as context.
5. The LLM generates a document-grounded response.

---

# Key Features

| Feature | Description |
|---|---|
| PDF Upload | Upload a PDF directly through the web interface |
| Text Extraction | Extract document text using `pypdf` |
| Semantic Search | Find relevant content using vector similarity |
| RAG | Retrieve context before generating an answer |
| Multiple Questions | Ask multiple questions about the same PDF |
| Conversation History | Keep previous questions and answers visible |
| Document Detection | Detect when a different PDF is uploaded |
| Grounded Responses | Restrict answers to retrieved document context |
| Clear Conversation | Reset the current question-answer history |
| Cloud Deployment | Deployed using Streamlit Community Cloud |

---

# Architecture

## High-Level Architecture

```text
                         ┌─────────────────┐
                         │      User       │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Streamlit UI   │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │   PDF Upload    │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  pypdf Parser   │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │ Text Chunking   │
                         │ 1200 / 250      │
                         └────────┬────────┘
                                  │
                                  ▼
                       ┌──────────────────────┐
                       │ Sentence Transformer │
                       │ all-MiniLM-L6-v2     │
                       └──────────┬───────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  FAISS Index    │
                         └────────┬────────┘
                                  │
                                  │
                     User Question
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Question Embedding │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │  Similarity Search │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Top 5 Relevant      │
                       │ Document Chunks     │
                       └──────────┬──────────┘
                                  │
                                  ▼
                       ┌─────────────────────┐
                       │ Context + Question │
                       └──────────┬──────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │    Groq LLM     │
                         │ gpt-oss-20b     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Final Answer   │
                         └─────────────────┘
```

---

# RAG Pipeline

DocQuery AI separates the system into two main stages.

## 1. Document Indexing

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Text Chunking
 │
 ▼
Embedding Generation
 │
 ▼
FAISS Vector Index
```

The document is processed when it is uploaded.

The current configuration uses:

```text
Chunk Size : 1200 characters
Overlap    : 250 characters
Embedding  : all-MiniLM-L6-v2
Index      : FAISS IndexFlatL2
```

---

## 2. Question Answering

```text
Question
   │
   ▼
Question Embedding
   │
   ▼
FAISS Similarity Search
   │
   ▼
Top 5 Relevant Chunks
   │
   ▼
Context Construction
   │
   ▼
Groq LLM
   │
   ▼
Document-Grounded Answer
```

The retrieved context is passed to the LLM together with the user's question.

The model is instructed not to introduce information that is not present in the retrieved document context.

---

# Technology Stack

### Frontend

**Streamlit**

Used to build the complete interactive web interface.

### PDF Processing

**pypdf**

Used to read PDF files and extract text from individual pages.

### Embeddings

**Sentence Transformers**

Embedding model:

```text
all-MiniLM-L6-v2
```

The model converts document chunks and user questions into numerical vectors.

### Vector Search

**FAISS**

FAISS is used to store the document embeddings and perform similarity search.

### Large Language Model

**Groq API**

Current model:

```text
openai/gpt-oss-20b
```

### Deployment

**Streamlit Community Cloud**

The application is connected directly to the GitHub repository for deployment.

---

# Project Structure

```text
DocQuery-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
└── assets/
    └── dashboard.png
```

### `app.py`

Contains the main application logic:

- Streamlit interface
- PDF processing
- Text chunking
- Embedding generation
- FAISS indexing
- Semantic retrieval
- Groq integration
- Conversation management
- Document change detection

### `requirements.txt`

Contains the Python packages required by the application.

```text
streamlit
pypdf
sentence-transformers
faiss-cpu
groq
```

### `assets/dashboard.png`

Optional project screenshot used in the README.

---

# Multiple Question Support

One of the important features of the application is that the same document can be queried multiple times.

For example:

```text
PDF
 │
 ├── What are the achievements?
 │
 ├── What projects are mentioned?
 │
 ├── Which technologies were used?
 │
 ├── What certifications are listed?
 │
 └── What internships are mentioned?
```

The document does not need to be uploaded again for each question.

Once processed, the chunks and FAISS index remain available in the current Streamlit session.

---

# Conversation Flow

The application maintains a lightweight conversation history.

```text
Question 1
    ↓
Answer 1

Question 2
    ↓
Answer 2

Question 3
    ↓
Answer 3
```

The user can review previous questions and answers while continuing to ask new questions about the same document.

The conversation can be cleared using the **Clear Conversation** option.

---

# Document Change Detection

DocQuery AI generates a hash-based identifier from the uploaded PDF content.

This helps distinguish between:

- The same document
- A newly uploaded document
- Different PDFs with the same filename

When a new document is detected, the application:

```text
New PDF
   ↓
Process PDF
   ↓
Create Chunks
   ↓
Generate Embeddings
   ↓
Build FAISS Index
   ↓
Clear Previous Conversation
```

---

# Prompt Grounding

The LLM receives instructions to answer using only the retrieved context.

The basic generation flow is:

```text
Retrieved Context
       +
User Question
       ↓
     Prompt
       ↓
    Groq LLM
       ↓
     Answer
```

If the retrieved context does not contain the requested information, the application instructs the model to return:

> I could not find this information in the uploaded PDF.

This is intended to reduce unsupported responses.

---

# Security

The Groq API key is not stored directly inside the source code.

The application reads the key through an environment variable:

```python
api_key = os.environ.get("GROQ_API_KEY")
```

For Streamlit Community Cloud, the key is configured using Streamlit Secrets.

The API key should never be committed to GitHub.

---

# Local Installation

## 1. Clone the repository

```bash
git clone https://github.com/fatimazaki1509/DocQuery-AI.git
cd DocQuery-AI
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure the API key

### Windows PowerShell

```powershell
$env:GROQ_API_KEY="your_api_key"
```

### Linux/macOS

```bash
export GROQ_API_KEY="your_api_key"
```

## 5. Start the application

```bash
streamlit run app.py
```

---

# Deployment

The application is deployed through **Streamlit Community Cloud**.

Deployment architecture:

```text
GitHub
  │
  ▼
Streamlit Community Cloud
  │
  ├── Install requirements
  ├── Load Secrets
  └── Run app.py
        │
        ▼
   Live Application
```

Repository changes can be committed to GitHub and reflected in the deployed application.

---

# Example Use Cases

## Resume Analysis

Upload a resume and ask:

```text
What are the candidate's major achievements?
```

```text
Which technical skills are mentioned?
```

```text
What projects has the candidate worked on?
```

## Research Papers

Upload a research paper and ask:

```text
What problem does the paper address?
```

```text
What methodology was used?
```

```text
What are the major findings?
```

## Academic Notes

Upload study material and ask:

```text
Explain the concept discussed in the document.
```

```text
List the important points from this chapter.
```

## Reports

Upload a business or technical report and ask:

```text
What are the major findings?
```

```text
What recommendations are mentioned?
```

---

# Future Improvements

Possible extensions include:

- OCR support for scanned PDFs
- Multi-document querying
- Persistent vector databases
- Page-level source citations
- Semantic chunking

---

# Project Goals

The project focuses on demonstrating the core components of a practical RAG application:

```text
Document Processing
        +
Semantic Embeddings
        +
Vector Search
        +
Context Retrieval
        +
LLM Generation
        =
Document Question Answering
```

The implementation keeps the architecture relatively simple while covering the major stages involved in building a retrieval-based LLM application.

---

# Author

## Fatima Zaki

**B.Tech — Computer Science and Engineering**

Nagpur, India

[GitHub](https://github.com/fatimazaki1509) · [LinkedIn](https://www.linkedin.com/in/fatima-zaki/)

---

# License

This project is intended for educational, portfolio, and demonstration purposes.
