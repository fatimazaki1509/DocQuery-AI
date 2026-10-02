import os
import streamlit as st
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="DocQuery AI",
    page_icon="📄",
    layout="centered"
)

st.title("📄 DocQuery AI")
st.write("Upload a PDF and ask questions about its content.")


# -----------------------------
# Groq API
# -----------------------------
api_key = os.environ.get("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is not configured.")
    st.stop()

client = Groq(api_key=api_key)


# -----------------------------
# Load Embedding Model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# -----------------------------
# Create Chunks
# -----------------------------
def create_chunks(text, chunk_size=1200, overlap=250):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# -----------------------------
# PDF Processing
# -----------------------------
def process_pdf(uploaded_file):

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    chunks = create_chunks(text)

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(
        embeddings.astype("float32")
    )

    return chunks, index


# -----------------------------
# Retrieve Relevant Chunks
# -----------------------------
def retrieve_chunks(question, chunks, index, k=5):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for i in range(len(indices[0])):

        chunk_index = indices[0][i]

        if chunk_index < len(chunks):

            results.append(
                chunks[chunk_index]
            )

    return results


# -----------------------------
# Ask Question using Groq
# -----------------------------
def ask_question(question, chunks, index):

    results = retrieve_chunks(
        question,
        chunks,
        index
    )

    context = "\n\n".join(results)

    prompt = f"""
You are an AI assistant that answers questions about a user's PDF.

Answer the question ONLY using the information provided in the context.

If the answer is not available in the context, say:

"I could not find this information in the uploaded PDF."

Do not make up information.

Context:
{context}

Question:
{question}

Answer clearly and concisely:
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


# -----------------------------
# PDF Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file
        != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):

            chunks, index = process_pdf(
                uploaded_file
            )

            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.processed_file = uploaded_file.name

        st.success(
            f"PDF processed successfully! "
            f"{len(chunks)} chunks created."
        )


# -----------------------------
# Question Section
# -----------------------------
if "chunks" in st.session_state:

    question = st.text_input(
        "Ask a question about your PDF:"
    )

    if st.button("🔍 Ask Question"):

        if question.strip():

            with st.spinner("Thinking..."):

                answer = ask_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.index
                )

            st.subheader("Answer")

            st.write(answer)

        else:

            st.warning(
                "Please enter a question."
            )
