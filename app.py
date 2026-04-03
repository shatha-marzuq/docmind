import streamlit as st
import os
from dotenv import load_dotenv

from core.document_processor import process_uploaded_file, chunk_documents
from core.embeddings import get_embeddings, add_documents_to_store, clear_vector_store
from core.rag_chain import answer_question
from core.hybrid_search import HybridRetriever

load_dotenv()

st.set_page_config(
    page_title="DocMind",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500&display=swap');

:root {
    --bg:           #ffffff;
    --surface:      #faf9f6;
    --ink:          #1a1a18;
    --ink2:         #5a5a54;
    --ink3:         #9a9a92;
    --line:         #e8e6e0;
    --line2:        #f0ede6;
    --teal:         #0f6e56;
    --teal-bg:      #e1f5ee;
    --accent:       #1D9E75;
    --user-bg:      #1a1a18;
    --user-text:    #f5f4f0;
    --r-sm:         8px;
    --radius:       16px;
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
[data-testid="stSidebarCollapseButton"],
#MainMenu, footer,
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stMainBlockContainer"] {
    padding: 1rem 1rem 2rem !important;
    max-width: 100% !important;
}

/* ── Left panel (settings) ── */
.panel {
    background: var(--surface);
    border: 0.5px solid var(--line);
    border-radius: var(--radius);
    padding: 1.4rem 1.2rem;
    height: fit-content;
    position: sticky;
    top: 1rem;
    margin-bottom: 0.5rem !important; /* أضف هذا السطر أو عدله */
}

/* ── Buttons ── */
.stButton > button {
    background: var(--bg) !important;
    color: var(--ink2) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: var(--r-sm) !important;
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
    background: var(--bg) !important;
    border: 1.5px dashed var(--line) !important;
    border-radius: var(--radius) !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent) !important;
    background: var(--teal-bg) !important;
}
[data-testid="stFileUploader"] label {
    font-size: 0.78rem !important;
    color: var(--ink3) !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: var(--ink3) !important;
    font-size: 0.75rem !important;
}

/* ── Upload strip ── */
.upload-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--teal-bg);
    border: 0.5px solid rgba(29,158,117,0.2);
    border-radius: var(--r-sm);
    padding: 7px 10px;
    margin-bottom: 6px;
    font-size: 0.74rem;
    color: var(--teal);
    overflow: hidden;
}
.ready-dot {
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.ready-badge {
    font-size: 9px;
    padding: 1px 6px;
    background: rgba(29,158,117,0.15);
    color: var(--teal);
    border-radius: 20px;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-left: auto;
    flex-shrink: 0;
}

/* ── Chat header ── */
.chat-header {
    display: flex;
    align-items: center;
    padding: 0 0 1rem;
    border-bottom: 0.5px solid var(--line);
    margin-bottom: 1.2rem;
}
.chat-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--ink);
    display: flex;
    align-items: center;
    gap: 8px;
}
.chat-doc-tag {
    font-size: 0.72rem;
    font-weight: 400;
    color: var(--ink3);
    background: var(--line2);
    padding: 2px 8px;
    border-radius: 20px;
}

/* ── Chat bubbles ── */
.msg-wrap-user {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    margin: 0.7rem 0;
}
.msg-wrap-ai {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    margin: 0.7rem 0;
}
.msg-role {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--ink3);
    margin-bottom: 5px;
}
.bubble-user {
    background: var(--user-bg);
    color: var(--user-text);
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1rem;
    max-width: 80%;
    font-size: 0.88rem;
    line-height: 1.65;
    font-weight: 300;
}
.bubble-ai {
    background: var(--surface);
    border: 0.5px solid var(--line);
    border-radius: 16px 16px 16px 4px;
    padding: 0.85rem 1rem;
    max-width: 88%;
    font-size: 0.88rem;
    line-height: 1.7;
    color: var(--ink);
}

/* ── Citations ── */
.cit-wrap {
    margin-top: 0.6rem;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.cit-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 9px;
    background: var(--teal-bg);
    border: 0.5px solid rgba(29,158,117,0.2);
    border-radius: 6px;
    font-size: 0.72rem;
    color: var(--teal);
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
    background: var(--surface) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: 14px !important;
    font-size: 0.88rem !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: 14px !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--line2) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: var(--r-sm) !important;
    font-size: 0.78rem !important;
    color: var(--ink) !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 0.5px solid var(--line) !important;
    border-radius: var(--r-sm) !important;
    background: var(--bg) !important;
}

/* ── Error ── */
.err-box {
    background: #fff5f5;
    border: 0.5px solid #fca5a5;
    border-radius: var(--r-sm);
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

hr { border-color: var(--line) !important; }

.section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 1.2px;
    color: #9a9a92;
    text-transform: uppercase;
    margin-bottom: 8px;
    display: block;
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

# ── Layout: two columns ───────────────────────────────────────────────────
top_k = 5
col_left, col_right = st.columns([1, 2.8], gap="medium")

# ── Left column: settings panel ───────────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="panel">
        <div style="font-family:serif;font-size:1.5rem;color:#1a1a18;
                    display:flex;align-items:center;gap:9px;margin-bottom:4px;">
            <span style="width:8px;height:8px;background:#1D9E75;border-radius:50%;
                         display:inline-block;flex-shrink:0;"></span>
            DocMind
        </div>
        <div style="font-size:0.72rem;color:#9a9a92;font-weight:300;margin-bottom:1.4rem;">
            Chat with your documents intelligently
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="section-label">Search Mode</span>', unsafe_allow_html=True)
    search_mode = st.selectbox("", ["Hybrid", "Semantic", "Keyword"], label_visibility="collapsed")

    st.markdown("<hr style='border:none;border-top:0.5px solid #e8e6e0;margin:1rem 0'>", unsafe_allow_html=True)

    with st.expander("Advanced Settings"):
        depth = st.selectbox(
            "Search Depth",
            ["Fast — quick specific questions", "Balanced — most questions", "Deep — summaries & analysis"],
            index=1,
        )
        if depth.startswith("Fast"):
            top_k = 3
        elif depth.startswith("Balanced"):
            top_k = 5
        else:
            top_k = 10

    st.markdown("<hr style='border:none;border-top:0.5px solid #e8e6e0;margin:1rem 0'>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">Document</span>', unsafe_allow_html=True)

    if not st.session_state.documents_loaded:
        uploaded_files = st.file_uploader(
            "Upload",
            type=["pdf", "txt", "docx", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            if st.button("Process Documents", use_container_width=True):
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
        for name in st.session_state.uploaded_files_names:
            st.markdown(
                f'<div class="upload-strip">'
                f'<span class="ready-dot"></span>'
                f'<span style="font-weight:500;color:#0f6e56;flex:1;overflow:hidden;'
                f'text-overflow:ellipsis;white-space:nowrap;">{name}</span>'
                f'<span class="ready-badge">READY</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("<hr style='border:none;border-top:0.5px solid #e8e6e0;margin:1rem 0'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with c2:
        if st.button("Reset All", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            clear_vector_store()
            st.rerun()

# ── Right column: chat ────────────────────────────────────────────────────
with col_right:
    doc_tag = ""
    if st.session_state.uploaded_files_names:
        doc_tag = f'<span class="chat-doc-tag">{st.session_state.uploaded_files_names[0]}</span>'

    st.markdown(
        f'<div class="chat-header">'
        f'<div class="chat-title">Conversation {doc_tag}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="msg-wrap-user">'
                f'<span class="msg-role">You</span>'
                f'<div class="bubble-user">{msg["content"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            cit_html = ""
            if msg.get("citations"):
                items = ""
                for cit in msg["citations"]:
                    page_info = f" · p.{cit['page']}" if cit.get("page") else ""
                    items += (
                        f'<div class="cit-item">'
                        f'<span class="cit-src">{cit["source"]}{page_info}</span>'
                        f'</div>'
                    )
                cit_html = f'<div class="cit-wrap">{items}</div>'
            st.markdown(
                f'<div class="msg-wrap-ai">'
                f'<span class="msg-role">DocMind</span>'
                f'<div class="bubble-ai">{msg["content"]}{cit_html}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    placeholder = "Ask anything about your documents…" if st.session_state.documents_loaded else "Upload a document to start…"
    if question := st.chat_input(placeholder, disabled=not st.session_state.documents_loaded):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Searching…"):
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
