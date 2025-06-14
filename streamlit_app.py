import os
import tempfile
import streamlit as st
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="\ud83d\udcc4 MCP – Document Q&A", page_icon="\ud83d\udcc4")
st.title("\ud83d\udcc4 Memory‑Controlled Pipeline (MCP) – Document Q&A")

# --------------------------------------------------
# Session‑state initialisation
# --------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# Sidebar – configuration
# --------------------------------------------------
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="You can also set OPENAI_API_KEY in Streamlit secrets."
    )
    chunk_size = st.number_input("Chunk size", 256, 4096, 1024, 128)
    chunk_overlap = st.number_input("Chunk overlap", 0, 1024, 128, 64)
    if st.button("\u274c Clear vector store"):
        st.session_state.vector_store = None
        st.success("Vector store cleared (disk & memory).")

# --------------------------------------------------
# Helper – load a single document
# --------------------------------------------------
def load_document(uploaded_file):
    suffix = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    if suffix == "pdf":
        loader = PyPDFLoader(path)
    elif suffix == "docx":
        loader = Docx2txtLoader(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
    return loader.load()

# --------------------------------------------------
# Upload & process documents
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload documents (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Add as many as you like – they’ll be embedded & stored in Chroma."
)

if uploaded_files and api_key:
    docs = []
    for uf in uploaded_files:
        docs.extend(load_document(uf))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_size),
        chunk_overlap=int(chunk_overlap)
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-large"
    )

    # Initialise or extend the store
    if st.session_state.vector_store is None:
        st.session_state.vector_store = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory="./chroma_db"
        )
    else:
        st.session_state.vector_store.add_documents(chunks)

    st.success(f"\u2705 {len(chunks)} chunks embedded & added to the vector store.")

# --------------------------------------------------
# Optional: lightweight API mode via URL param
# e.g. https://your-app-url.streamlit.app/?query=What+is+...&api=1
# --------------------------------------------------
params = st.experimental_get_query_params()
if "query" in params and api_key and st.session_state.vector_store is not None:
    q_param = params["query"][0]
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        model_name="gpt-4o-mini"
    )
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = chain({"question": q_param, "chat_history": []})
    st.json({"answer": result["answer"]})
    st.stop()  # Don’t render the rest of the UI in API mode

# --------------------------------------------------
# Interactive Q&A
# --------------------------------------------------
query = st.text_input("Ask a question about your documents:")

if query and api_key and st.session_state.vector_store is not None:
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        model_name="gpt-4o-mini"
    )
    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = qa_chain({"question": query, "chat_history": st.session_state.chat_history})
    answer = result["answer"]

    # Save chat history
    st.session_state.chat_history.append((query, answer))

    # Display
    st.markdown(f"**Answer:** {answer}")
    with st.expander("\ud83d\udd0d Retrieved context"):
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}**\n```\n{doc.page_content}\n```")

# --------------------------------------------------
# Chat history display
# --------------------------------------------------
if st.session_state.chat_history:
    st.header("\ud83d\udd5d\ufe0f Chat History")
    for q, a in reversed(st.session_state.chat_history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
