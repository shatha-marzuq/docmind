import streamlit as st
import os
from dotenv import load_dotenv

from core.document_processor import process_uploaded_file, chunk_documents
from core.embeddings import get_embeddings, add_documents_to_store, clear_vector_store
from core.rag_chain import answer_question
from core.hybrid_search import HybridRetriever

load_dotenv()

st.set_page_config(page_title="DocMind", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500&display=swap');

:root {
    --bg:         #faf9f6;
    --white:      #ffffff;
    --ink:        #1a1a18;
    --ink2:       #5a5a54;
    --ink3:       #9a9a92;
    --line:       #e8e6e0;
    --line2:      #f0ede6;
    --teal:       #0f6e56;
    --teal-bg:    #e1f5ee;
    --accent:     #1D9E75;
    --accent-light: #c0f0df;
    --user-bg:    #1a1a18;
    --user-text:  #f5f4f0;
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"],
[data-testid="stSidebar"],
[data-testid="collapsedControl"],
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

/* ── Header ── */
.doc-header {
    padding: 2rem 0 1.2rem;
    border-bottom: 0.5px solid var(--line);
    margin-bottom: 1.5rem;
}
.doc-header h1 {
    font-family: 'Instrument Serif', serif !important;
    font-size: 2rem;
    font-weight: 400;
    color: var(--ink);
    letter-spacing: -0.5px;
    margin: 0 0 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.brand-dot {
    width: 8px; height: 8px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
}
.doc-header p {
    font-size: 0.82rem;
    color: var(--ink3);
    font-weight: 300;
    margin: 0;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--white) !important;
    color: var(--ink2) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.4rem 1rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all .15s !important;
}
.stButton > button:hover {
    border-color: var(--ink3) !important;
    color: var(--ink) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--white) !important;
    border: 1.5px dashed var(--line) !important;
    border-radius: 16px !important;
    transition: all 0.2s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--teal-bg) !important;
}
[data-testid="stFileUploader"] label {
    font-size: 0.82rem !important;
    color: var(--ink3) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--ink3) !important;
}

/* ── Upload strip ── */
.upload-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--teal-bg);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 1rem;
    font-size: 0.75rem;
    color: var(--teal);
    font-weight: 400;
}
.ready-dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: var(--user-bg);
    color: var(--user-text);
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 72%;
    font-size: 0.88rem;
    line-height: 1.6;
    width: fit-content;
}
.bubble-ai {
    background: var(--white);
    border: 0.5px solid var(--line);
    border-radius: 16px 16px 16px 4px;
    padding: 0.85rem 1rem;
    margin: 0.5rem auto 0.5rem 0;
    max-width: 82%;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--ink);
    width: fit-content;
}
.msg-role {
    font-size: 0.7rem;
    color: var(--ink3);
    font-weight: 500;
    letter-spacing: 0.3px;
    margin-bottom: 4px;
    display: block;
}

/* ── Citations ── */
.cit-wrap {
    margin-top: 0.6rem;
    border-top: 0.5px solid var(--line);
    padding-top: 0.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.cit-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 8px;
    background: var(--teal-bg);
    border-radius: 20px;
    font-size: 0.72rem;
    color: var(--teal);
    font-weight: 400;
}
.cit-src {
    font-weight: 500;
    color: var(--teal);
    font-size: 0.7rem;
    font-family: monospace !important;
}
.highlight-keyword {
    background: #d1fae5;
    border-radius: 2px;
    padding: 0 2px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: var(--white) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: 12px !important;
    font-size: 0.88rem !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] {
    background: var(--white) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: 12px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--line2) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: 8px !important;
    font-size: 0.8rem !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 0.5px solid var(--line) !important;
    border-radius: 8px !important;
    background: var(--white) !important;
}

/* ── Error ── */
.err-box {
    background: #fff5f5;
    border: 0.5px solid #fca5a5;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #991b1b;
    margin: 0.5rem 0;
}

/* ── Bottom ── */
[data-testid="stBottom"] > div,
.stChatFloatingInputContainer {
    background: var(--bg) !important;
}

hr {
    border-color: var(--line) !important;
}
</style>
""", unsafe_allow_html=True)
# ── Session state ─────────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "documents_loaded": False,
    "uploaded_files_names": [],
    "embeddings": None,
    "vector_store": None,
    "all_chunks": [],
    "hybrid_retriever": None,
    "search_mode": "Hybrid",
    "top_k": 5,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def highlight_keywords(text, query):
    import re
    result = text
    for word in query.lower().split():
        if len(word) > 3:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            result = pattern.sub(lambda m: f'<span class="highlight-keyword">{m.group()}</span>', result)
    return result

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="doc-header">
    <div>
        <h1>DocMind</h1>
        <p>Chat with your documents intelligently</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Top controls ──────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    search_mode = st.selectbox("", ["Hybrid", "Semantic", "Keyword"], label_visibility="collapsed")
with col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
with col3:
    if st.button("Reset All", use_container_width=True):
        for k, v in defaults.items():
            st.session_state[k] = v
        clear_vector_store()
        st.rerun()

with st.expander("Advanced Settings"):
    depth = st.selectbox(
        "Search Depth",
        [
            "Fast — quick specific questions",
            "Balanced — most questions",
            "Deep — summaries & analysis",
        ],
        index=1,
    )
    if depth.startswith("Fast"):
        top_k = 3
    elif depth.startswith("Balanced"):
        top_k = 5
    else:
        top_k = 10

st.markdown("<hr style='border:none;border-top:1px solid #C8D0E8;margin:0.5rem 0 1rem'>", unsafe_allow_html=True)
# ── Upload ────────────────────────────────────────────────────────────────
if not st.session_state.documents_loaded:
    uploaded_files = st.file_uploader(
        "Upload your files",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Processing..."):
                try:
                    all_chunks = []
                    for file in uploaded_files:
                        docs = process_uploaded_file(file)
                        chunks = chunk_documents(docs)
                        all_chunks.extend(chunks)
                    if st.session_state.embeddings is None:
                        st.session_state.embeddings = get_embeddings()
                    clear_vector_store()
                    st.session_state.vector_store = add_documents_to_store(all_chunks, st.session_state.embeddings)
                    st.session_state.hybrid_retriever = HybridRetriever(st.session_state.vector_store, all_chunks)
                    st.session_state.all_chunks = all_chunks
                    st.session_state.documents_loaded = True
                    st.session_state.uploaded_files_names = [f.name for f in uploaded_files]
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.markdown(f'<div class="err-box">{e}</div>', unsafe_allow_html=True)
else:
    pills = "".join([f'<span style="background:#E5E7EB;border-radius:6px;padding:2px 8px;font-size:0.73rem;margin:2px;">{n}</span>' for n in st.session_state.uploaded_files_names])
    st.markdown(f'<div class="upload-strip"><span class="ready-dot"></span>&nbsp;Ready &nbsp;{pills}</div>', unsafe_allow_html=True)

# ── Chat ──────────────────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        cit_html = ""
        if msg.get("citations"):
            items = ""
            for cit in msg["citations"]:
                page_info = f" · p.{cit['page']}" if cit.get("page") else ""
                hl = highlight_keywords(cit["content"], msg.get("query", ""))
                items += f'<div class="cit-item"><div class="cit-src">{cit["source"]}{page_info}</div>{hl}</div>'
            cit_html = f'<div class="cit-wrap">{items}</div>'
        st.markdown(f'<div class="bubble-ai">{msg["content"]}{cit_html}</div>', unsafe_allow_html=True)

placeholder = "Ask anything about your documents..." if st.session_state.documents_loaded else "Upload a document to start..."
if question := st.chat_input(placeholder, disabled=not st.session_state.documents_loaded):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.spinner("Searching..."):
        try:
            if search_mode == "Hybrid":
                docs_with_scores = st.session_state.hybrid_retriever.retrieve(question, top_k)
            elif search_mode == "Semantic":
                docs_with_scores = st.session_state.vector_store.similarity_search_with_score(question, k=top_k)
            else:
                from core.hybrid_search import BM25Retriever
                bm25 = BM25Retriever(st.session_state.all_chunks)
                docs_with_scores = bm25.search(question, k=top_k)
            result = answer_question(question, docs_with_scores)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"],
                "query": question,
            })
            st.rerun()
        except Exception as e:
            st.markdown(f'<div class="err-box">{e}</div>', unsafe_allow_html=True)
