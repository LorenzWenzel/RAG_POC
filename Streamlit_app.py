import sys
from pathlib import Path
from datetime import datetime
import streamlit as st
import re

# --- Import aus deinem RAG-Code ---
from RAG_POC import run_retrieval_test, answer_query, filter_chunks_iterative_exact, top_k_chunks
from UploadAnswer import run_retrieval_test_upload, answer_query_upload
from chunk_post_process import apply_per_doc_cap

# --- Seite früh konfigurieren ---
st.set_page_config(page_title="RAG Query Chat (PoC)", page_icon="💬", layout="centered")

# Rechts-Spalte sticky machen (nur Desktop)
st.markdown("""
<style>
@media (min-width: 768px){
  /* zweite Spalte (rechts) kleben lassen */
  div[data-testid="column"]:nth-of-type(2) > div {
    position: sticky;
    top: 1rem;
    align-self: flex-start;
  }
}
</style>
""", unsafe_allow_html=True)



# ----------------------------------
# Pfade & Settings
# ----------------------------------
BASE_DIR       = Path(".")
QUERIES_DIR    = BASE_DIR / "queries"
RETRIEVED_DIR  = BASE_DIR / "Retrieved"   # wird weiter von run_retrieval_test genutzt (für Logs/Debug)
EXTERNAL_FILES_DIR= BASE_DIR / "ExternalFiles" 
QUERIES_DIR.mkdir(exist_ok=True)
RETRIEVED_DIR.mkdir(exist_ok=True)
EXTERNAL_FILES_DIR.mkdir(exist_ok=True)

LAST_QUERY_FILE = QUERIES_DIR / "last_query.txt"
HISTORY_FILE    = QUERIES_DIR / "history.txt"

# Anzahl der chunks die geholt werden zu beginn
Frist_K = 22
# Maximale Anzahl an chunks die in das llm gehen
TOP_K = 3
#Anzahl der erlaubten chunks pro dokument 
PER_DOC_CAP=4




# ----------------------------------
# Chat-State
# ----------------------------------
    


def _sanitize_filename(name: str) -> str:
    # einfache Sanitisierung, behält Endungen bei
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
        # Datei wird überschrieben, falls sie schon existiert
        target.write_bytes(uf.read())
        saved.append(target)
    return saved







# ====== Session-State ======
if "mode" not in st.session_state:
    st.session_state.mode = "RAG-Chat"

# separater Chat-Verlauf pro Modus
if "messages_RAG-Chat" not in st.session_state:
    st.session_state["messages_RAG-Chat"] = []
if "messages_Upload-Chat" not in st.session_state:
    st.session_state["messages_Upload-Chat"] = []

# Kontext-Dateiliste für Upload-Chat (nur Dateinamen; später durch doc_ids ersetzbar)
if "context_doc_ids" not in st.session_state:
    st.session_state["context_doc_ids"] = []    

# ====== Sidebar ======
with st.sidebar:
    st.subheader("Modus")
    mode = st.radio(
        "Wähle den Chat:",
        options=("RAG-Chat", "Upload-Chat"),
        index=0 if st.session_state.mode == "RAG-Chat" else 1,
    )
    st.session_state.mode = mode

    st.divider()
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

# ====== Header (modusabhängig) ======
st.title("💬 RAG Query Chat (PoC)")
if mode == "RAG-Chat":
    st.caption("Frage eingeben → Retrieval läuft → Top-K Treffer werden angezeigt → Antwort wird mit Quellen erstellt.")
else:
    st.caption("Modus im Sidebar wählen. Upload-Chat erlaubt zusätzlich Datei-Uploads (Speicher: ./ExternalFiles).")

with st.expander("🔧 Diagnose", expanded=False):
    st.write(f"History-Datei: `{HISTORY_FILE.resolve()}`")

# ====== Upload-UI (nur im Upload-Chat) ======
uploaded_paths = []
if mode == "Upload-Chat":
    st.markdown("### 📎 Dateien hochladen")
    up = st.file_uploader(
        "Dateien (PDF, Bilder, Text) auswählen – sie werden in `./ExternalFiles` gespeichert.",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "txt", "md"],
        accept_multiple_files=True,
        help="Die Dateien werden nur gespeichert. (Kein automatisches Indexieren in diesem Schritt.)",
    )
    if up:
        saved = save_uploaded_files(up)
        if saved:
            st.success(f"{len(saved)} Datei(en) gespeichert.")
            uploaded_paths = saved
            with st.expander("Gespeicherte Dateien"):
                for p in saved:
                    st.write(f"• {p.name}  \t({p.stat().st_size} Bytes)")

    st.divider()

    # --- Auswahlfeld: welche Dateien sollen in den (UI-)Kontext? ---
    st.markdown("### 🧩 Kontext-Steuerung")
    # Verfügbare Dateien (aktuell im Ordner)
    available_files = sorted([p.name for p in EXTERNAL_FILES_DIR.iterdir() if p.is_file()])

    # Multi-Select für Auswahl (nur Frontend; keine Abfrage-Logik hier)
    selection = st.multiselect(
        "Dateien auswählen, die dem Kontext hinzugefügt werden sollen:",
        options=available_files,
        default=[],
        placeholder="Wähle 1–10 Dateien aus",
        help="Nur UI-State. Die eigentliche Filter-Logik fügst du später an."
    )

    cols = st.columns([1,1,2])
    with cols[0]:
        if st.button("Kontext hinzufügen", use_container_width=True, type="primary", disabled=(len(selection) == 0)):
            # Merge ohne Duplikate
            existing = set(st.session_state["context_doc_ids"])
            to_add   = [s for s in selection if s not in existing]
            st.session_state["context_doc_ids"].extend(to_add)
            st.success(f"{len(to_add)} Datei(en) zum Kontext hinzugefügt.")

    with cols[1]:
        if st.button("Kontext leeren", use_container_width=True):
            st.session_state["context_doc_ids"].clear()
            st.info("Kontext geleert.")

    # --- Anzeige des aktuellen Kontextes ---
    st.markdown("#### Kontext:")
    if not st.session_state["context_doc_ids"]:
        st.caption("*(Kein Kontext gesetzt – es werden später alle Dokumente ignoriert/Standardverhalten.)*")
    else:
        # hübsche „Chips“-Darstellung
        chip_cols = st.columns(3)
        for i, fname in enumerate(st.session_state["context_doc_ids"]):
            with chip_cols[i % 3]:
                st.code(fname, language=None)

    st.caption("Hinweis: Aktuell nur UI. Die Filter-Logik (z. B. `doc_id in context`) fügst du später an.")




# ====== Chat-Verlauf anzeigen (modusabhängig) ======
message_key = "messages_RAG-Chat" if mode == "RAG-Chat" else "messages_Upload-Chat"
for msg in st.session_state[message_key]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====== Eingabe ======
user_input = st.chat_input("Frage eingeben und Enter drücken …")

# ====== Verarbeitung ======
if user_input:
    # 1) User-Message anzeigen & loggen
    st.session_state[message_key].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    _save_last_query(user_input)
    st.toast("Query gespeichert.", icon="✅")

    # 2) Retrieval
    if mode == "Upload-Chat":
        with st.spinner("Suche relevante Textstellen (Top-K)…"):
            try:
                chunks = run_retrieval_test_upload(user_input, topK0=8, per_doc_cap=3)
            except Exception as e:
                chunks = []
                st.error(f"Retrieval-Fehler: {e}")

    else:
        # ===== Normaler RAG-Chat wie bisher =====
        with st.spinner("Suche relevante Textstellen (Top-K)…"):
            try:
                chunks = run_retrieval_test(user_input, topK0=Frist_K, per_doc_cap=PER_DOC_CAP)
            except Exception as e:
                chunks = []
                st.error(f"Retrieval-Fehler: {e}")

        # 3) Antwort
        if mode == "Upload-Chat":
            with st.spinner("Erzeuge Antwort …"):
                try:
                    #chunks auf k einheiten filtern
                    filtered_chunks = top_k_chunks(chunks, 6)
                    final_answer = answer_query_upload(user_input, filtered_chunks)
                except Exception as e:
                    final_answer = f"_LLM-Fehler beim Erzeugen der Antwort:_ `{e}`"

            with st.chat_message("assistant"):
                st.markdown(final_answer)

            st.session_state[message_key].append({"role": "assistant", "content": final_answer})
            _append_history("assistant", final_answer)
        else: 
            with st.spinner("Erzeuge Antwort …"):
                try:
                    # 1) clientseitig filtern
                    filtered_chunks = filter_chunks_iterative_exact(user_input, chunks)

                    #chunks auf k einheiten filtern
                    filtered_chunks = top_k_chunks(filtered_chunks, TOP_K)

                    # 2) Gefilterte Treffer anzeigen (vor der Antwort)
                    with st.chat_message("assistant"):
                        st.markdown(f"### 🔎 Gefilterte Treffer ({len(filtered_chunks)})")
                        if not filtered_chunks:
                            st.caption("*(Keine Filter-Treffer – verwende alle Chunks)*")
                        to_show = filtered_chunks or chunks  # Fallback auf alle Chunks

                        for i, r in enumerate(to_show, start=1):
                            st.markdown(f"**Treffer {i}** — {r.get('heading','')}")
                            st.caption(f"{r.get('doc_id')} · Chunk {int(r.get('chunk_no',0))}")

                            preview = (r.get("content") or "")
                            st.code(preview[:10000] + ("…" if len(preview) > 10000 else ""), language=None)

                            meta = r.get("meta", {})
                            if meta:
                                st.markdown("**Metadaten:**")
                                st.json(meta)
                            else:
                                st.caption("*(Keine Metadaten vorhanden)*")

                    # 3) Antwort mit (gefilterten) Chunks erzeugen
                    final_answer = answer_query(user_input, filtered_chunks or chunks)
                except Exception as e:
                    final_answer = f"_LLM-Fehler beim Erzeugen der Antwort:_ `{e}`"

            with st.chat_message("assistant"):
                st.markdown(final_answer)

            st.session_state[message_key].append({"role": "assistant", "content": final_answer})
            _append_history("assistant", final_answer)
