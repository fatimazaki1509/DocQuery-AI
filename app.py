```python
import os
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DocQuery AI",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* Main container */
    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .app-header {
        text-align: center;
        margin-bottom: 2.5rem;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 650;
        letter-spacing: -0.5px;
        margin-bottom: 0.35rem;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 1rem;
    }

    /* Section headings */
    .section-title {
        font-size: 0.85rem;
        font-weight: 650;
        letter-spacing: 0.7px;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 0.7rem;
        text-transform: uppercase;
    }

    /* Document card */
    .document-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        background: #ffffff;
        margin-bottom: 1.5rem;
    }

    .document-name {
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.2rem;
    }

    .document-status {
        color: #16a34a;
        font-size: 0.85rem;
    }

    /* Answer card */
    .answer-card {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem;
        background: #ffffff;
        margin-top: 0.7rem;
        margin-bottom: 1rem;
    }

    /* Divider */
    .divider {
        border-top: 1px solid #e5e7eb;
        margin: 2rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.78rem;
        margin-top: 3rem;
    }

    /* Buttons */
    div.stButton > button {
        border-radius: 7px;
        font-weight: 550;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">DocQuery AI</div>
        <div class="app-subtitle">
            Document Intelligence and Question Answering
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# GROQ CLIENT
# =========================================================

api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=api_key)


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# =========================================================
# SESSION STATE
# =========================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(
    text,
    chunk_size=1200,
    overlap=250
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# =========================================================
# PROCESS PDF
# =========================================================

def process_pdf(uploaded_file):

    reader = PdfReader(
        uploaded_file
    )

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    if not text.strip():

        return [], None

    chunks = create_chunks(text)

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    return chunks, index


# =========================================================
# RETRIEVE RELEVANT CHUNKS
# =========================================================

def retrieve_chunks(
    question,
    chunks,
    index,
    k=5
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for i in range(
        len(indices[0])
    ):

        chunk_index = indices[0][i]

        if 0 <= chunk_index < len(chunks):

            results.append(
                chunks[chunk_index]
            )

    return results


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(
    question,
    chunks,
    index
):

    relevant_chunks = retrieve_chunks(
        question,
        chunks,
        index,
        k=5
    )

    context = "\n\n".join(
        relevant_chunks
    )

    prompt = f"""
You are an AI assistant that answers questions about a user's uploaded PDF.

Use ONLY the information provided in the context.

If the answer cannot be found in the context, respond exactly with:

"I could not find this information in the uploaded PDF."

Do not make up information.
Do not use outside knowledge.
Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0
    )

    return response.choices[0].message.content


# =========================================================
# DOCUMENT UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">Document</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    label_visibility="collapsed"
)


# =========================================================
# PROCESS DOCUMENT
# =========================================================

if uploaded_file:

    # Process only if this is a NEW document
    if (
        st.session_state.processed_file
        != uploaded_file.name
    ):

        with st.spinner(
            "Processing document..."
        ):

            chunks, index = process_pdf(
                uploaded_file
            )

        if not chunks:

            st.error(
                "Could not extract readable text from this PDF."
            )

            st.stop()

        # Store processed document
        st.session_state.chunks = chunks
        st.session_state.index = index
        st.session_state.processed_file = uploaded_file.name

        # Clear old conversation
        st.session_state.messages = []

        st.success(
            f"Document processed successfully. "
            f"{len(chunks)} text sections indexed."
        )


# =========================================================
# SHOW DOCUMENT STATUS
# =========================================================

if st.session_state.processed_file:

    st.markdown(
        f"""
        <div class="document-card">
            <div class="document-name">
                {st.session_state.processed_file}
            </div>
            <div class="document-status">
                Document processed and ready for questions
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# CONVERSATION HISTORY
# =========================================================

if st.session_state.messages:

    st.markdown(
        '<div class="section-title">Conversation</div>',
        unsafe_allow_html=True
    )

    for message in st.session_state.messages:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div style="
                    margin-bottom:0.4rem;
                    font-weight:600;
                    color:#374151;
                ">
                    Question
                </div>

                <div style="
                    background:#f9fafb;
                    border:1px solid #e5e7eb;
                    border-radius:8px;
                    padding:0.9rem 1rem;
                    margin-bottom:1rem;
                ">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div style="
                    margin-bottom:0.4rem;
                    font-weight:600;
                    color:#374151;
                ">
                    Answer
                </div>

                <div class="answer-card">
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# QUESTION INPUT
# =========================================================

if st.session_state.chunks:

    st.markdown(
        '<div class="section-title">Ask about your document</div>',
        unsafe_allow_html=True
    )

    with st.form(
        key="question_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Question",
            placeholder="Ask a question about the uploaded document...",
            label_visibility="collapsed"
        )

        submitted = st.form_submit_button(
            "Ask Question",
            use_container_width=False
        )

    if submitted:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the document and generating an answer..."
            ):

                answer = ask_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.index
                )

            # Save conversation
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.rerun()


# =========================================================
# CLEAR CONVERSATION
# =========================================================

if st.session_state.messages:

    st.markdown(
        '<div class="divider"></div>',
        unsafe_allow_html=True
    )

    if st.button(
        "Clear Conversation"
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        DocQuery AI · Retrieval-Augmented Document Question Answering
    </div>
    """,
    unsafe_allow_html=True
)
```
