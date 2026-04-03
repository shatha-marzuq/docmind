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
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap');

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
    font-family: 'Instrument Serif', serif !important; /* تغيير الخط هنا */
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

.panel {
    background: var(--surface);
    border: 0.5px solid var(--line);
    border-radius: var(--radius);
    padding: 1.4rem 1.2rem;
    height: fit-content;
    position: sticky;
    top: 1rem;
}

.stButton > button {
    background: var(--bg) !important;
    color: var(--ink2) !important;
    border: 0.5px solid var(--line) !important;
    border-radius: var(--r-sm) !important;
    font-size: 0.78rem !important;
    font-weight: 400 !important;
    padding: 0.4rem 1rem !important;
    font-family: 'Instrument Serif', serif !important; /* تغيير الخط هنا أيضًا */
    transition: all .15s !important;
}
.stButton > button:hover {
    border-color: var(--ink3) !important;
    color: var(--ink) !important;
}

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

# ── Layout: two columns ───────────────────────────────────────────────────
top_k = 5
col_left, col_right = st.columns([1, 2.8], gap="medium")

# ── Left column: settings panel ───────────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="panel">
        <div style="font-family:'Instrument Serif', serif;font-size:1.5rem;color:#1a1a18;
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
