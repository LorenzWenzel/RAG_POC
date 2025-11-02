# common.py
import sys
from pathlib import Path
from datetime import datetime
import re
import streamlit as st

# --- Imports aus deinem RAG-Code (werden auf Seiten genutzt) ---
from RAG_POC import run_retrieval_test, answer_query, filter_chunks_iterative_exact, top_k_chunks
from UploadAnswer import (
    run_retrieval_test_upload,
    answer_query_upload,
    uploadFilesToIndex,
    delete_all_entries_in_index,
)
from chunk_post_process import apply_per_doc_cap  # falls benötigt

# ----------------------------------
# Pfade & Settings (shared)
# ----------------------------------
BASE_DIR           = Path(".")
QUERIES_DIR        = BASE_DIR / "queries"
RETRIEVED_DIR      = BASE_DIR / "Retrieved"
EXTERNAL_FILES_DIR = BASE_DIR / "ExternalFiles"
for d in (QUERIES_DIR, RETRIEVED_DIR, EXTERNAL_FILES_DIR):
    d.mkdir(exist_ok=True)

LAST_QUERY_FILE  = QUERIES_DIR / "last_query.txt"
HISTORY_FILE     = QUERIES_DIR / "history.txt"

# ----------------------------------
# Konfiguration (shared)
# ----------------------------------
FIRST_K     = 22   # (vormals "Frist_K")
TOP_K       = 3
PER_DOC_CAP = 4

# ----------------------------------
# Helpers (shared)
# ----------------------------------
def _sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w\-.]+", "_", name)
    return name[:150] or "upload"

def _append_history(role: str, content: str):
    ts = datetime.utcnow().isoformat() + "Z"
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{ts}\t{role}\t{content}\n")

def _save_last_query(q: str):
    LAST_QUERY_FILE.write_text(q, encoding="utf-8")
    _append_history("user", q)

def save_uploaded_files(uploaded_files):
    saved = []
    for uf in uploaded_files or []:
        raw_name = _sanitize_filename(uf.name)
        target = EXTERNAL_FILES_DIR / raw_name
        target.write_bytes(uf.read())  # überschreibt, falls vorhanden
        saved.append(target)
    return saved

def ensure_state():
    """Initialisiert alle benötigten Session-State-Keys."""
    if "messages_RAG-Chat" not in st.session_state:
        st.session_state["messages_RAG-Chat"] = []
    if "messages_Upload-Chat" not in st.session_state:
        st.session_state["messages_Upload-Chat"] = []
    if "context_doc_ids" not in st.session_state:
        st.session_state["context_doc_ids"] = []

def sidebar_status():
    with st.sidebar:
        st.subheader("Status")
        st.caption(f"Python `{sys.version.split()[0]}` · Streamlit `{st.__version__}`")
        st.write(f"Queries: `{QUERIES_DIR.resolve()}`")
        st.write(f"Retrieved: `{RETRIEVED_DIR.resolve()}`")
        st.write(f"Uploads: `{EXTERNAL_FILES_DIR.resolve()}`")
        if LAST_QUERY_FILE.exists():
            st.write("Letzte Query (gekürzt):")
            prev = LAST_QUERY_FILE.read_text(encoding="utf-8").strip()
            st.code(prev[:300] + ("…" if len(prev) > 300 else ""), language=None)
        else:
            st.write("Noch keine Query gespeichert.")

def render_messages(message_key: str):
    for msg in st.session_state[message_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# ----------------------------------
# UI-Controls: Upload & Delete-All Buttons (mit Action)
# ----------------------------------
def render_upload_controls():
    """
    Rendert die Buttons 'Upload' und 'Delete all' und führt bei Klick:
    - Upload:  uploadFilesToIndex()
    - Delete:  delete_all_entries_in_index(confirm=False)
    """
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Upload", use_container_width=True):
            with st.spinner("Indexiere Dateien aus ./ExternalFiles …"):
                try:
                    uploadFilesToIndex()
                    st.success("Uploads wurden in den Index geschrieben.")
                except Exception as e:
                    st.error(f"Fehler beim Upload-Indexing: {e}")

    # in render_upload_controls()
    with c2:
        if st.button("Delete all", use_container_width=True):
            with st.expander("Bestätigung", expanded=True):
                st.warning("Achtung: Dies löscht ALLE Einträge im Index.")
                col_ok, col_cancel = st.columns([1,1])
                with col_ok:
                    if st.button("Ja, alles löschen", use_container_width=True, key="confirm_delete_all"):
                        with st.spinner("Lösche alle Einträge aus dem Index …"):
                            try:
                                delete_all_entries_in_index(confirm=True)  # keine input()-Abfrage mehr!
                                st.success("Alle Einträge aus dem Index wurden gelöscht.")
                            except Exception as e:
                                st.error(f"Fehler beim Löschen: {e}")
                with col_cancel:
                    st.button("Abbrechen", use_container_width=True, key="cancel_delete_all")

