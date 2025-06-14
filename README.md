# 📄 MCP – Memory‑Controlled Pipeline (RAG) on Streamlit Cloud

A no‑install Retrieval‑Augmented Generation app that lets you upload documents, embed them with OpenAI `text-embedding-3-large`, store vectors in ChromaDB, and chat with GPT‑4o‑mini about your files.

## Quick Deploy

1. **Fork / clone** this repo or create a new GitHub repo and add `streamlit_app.py` & `requirements.txt`.
2. **Streamlit Community Cloud**
   * Go to <https://share.streamlit.io>
   * Click **New app** → connect your repo → select `streamlit_app.py`.
   * In **Secrets** add:
     ```
     OPENAI_API_KEY = "sk-..."
     ```
3. **Use the app**
   * Upload PDFs / DOCX / TXT
   * Ask questions
   * Optional API call:
     ```
     https://<your-app>.streamlit.app/?query=Summarise+section+4+of+the+policy&api=1
     ```

---

Vector store is persisted in `./chroma_db` (ephemeral on Streamlit Cloud). Swap `gpt-4o-mini` for any model you have access to.
