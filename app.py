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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
:root {
    --bg:      #F5F0E8;  /* خلفية بيج فاتح */
    --navy:    #2C2416;  /* نص بني داكن */
    --panel:   #EDE5D4;  /* خلفية البانلات */
    --border:  #D4C4A8;  /* حدود بيج داكن */
    --muted:   #8B7355;  /* نص ثانوي بني */
    --teal:    #A0845C;  /* أزرار بني ذهبي */
    --teal2:   #B8976A;  /* hover أفتح */
    --user-bg: #E8D9C0;  /* فقاعة المستخدم */
    --ai-bg:   #FAF7F2;  /* فقاعة الـ AI */
}

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main {
    background: var(--bg) !important;
    font-family: 'Inter', sans-serif !important;
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
    border-bottom: 1.5px solid var(--border);
    padding-bottom: 1rem;
    margin-bottom: 1.5rem;
    padding-top: 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.doc-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--navy);
    letter-spacing: -0.03em;
    margin: 0;
}
.doc-header p {
    font-size: 0.8rem;
    color: var(--muted);
    font-weight: 300;
    margin: 3px 0 0;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--teal) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    padding: 0.45rem 1.2rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: background .15s !important;
}
.stButton > button:hover { background: var(--teal2) !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: var(--panel) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] label {
    font-size: 0.85rem !important;
    color: var(--muted) !important;
}

/* ── Upload strip ── */
.upload-strip {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    font-size: 0.76rem;
    color: var(--muted);
}
.ready-dot {
    width: 7px; height: 7px;
    background: #5C9E6E;
    border-radius: 50%;
    display: inline-block;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: var(--user-bg);
    border-radius: 14px 14px 3px 14px;
    padding: 0.7rem 1rem;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 75%;
    font-size: 0.87rem;
    line-height: 1.55;
    color: var(--navy);
    width: fit-content;
}
.bubble-ai {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 3px;
    padding: 0.8rem 1rem;
    margin: 0.5rem auto 0.5rem 0;
    max-width: 85%;
    font-size: 0.87rem;
    line-height: 1.65;
    color: var(--navy);
    width: fit-content;
}

/* ── Citations ── */
.cit-wrap {
    margin-top: 0.6rem;
    border-top: 1px solid var(--border);
    padding-top: 0.5rem;
}
.cit-item {
    border-left: 2px solid var(--teal);
    padding: 0.4rem 0.7rem;
    margin: 0.3rem 0;
    font-size: 0.75rem;
    color: var(--muted);
    border-radius: 0 6px 6px 0;
    background: var(--panel);
}
.cit-src {
    font-weight: 600;
    color: var(--teal);
    font-size: 0.71rem;
    margin-bottom: 2px;
    font-family: monospace !important;
}
.highlight-keyword {
    background: #B2E8E2;
    border-radius: 2px;
    padding: 0 2px;
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    color: var(--navy) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: var(--panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    color: var(--navy) !important;
}

/* ── Error ── */
.err-box {
    background: #FDF0EE;
    border: 1px solid #E8B4AE;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    font-size: 0.82rem;
    color: #8B2019;
    margin: 0.5rem 0;
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
    pills = "".join([f'<span style="background:#D6DCF5;border-radius:6px;padding:2px 8px;font-size:0.73rem;margin:2px;">{n}</span>' for n in st.session_state.uploaded_files_names])
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
