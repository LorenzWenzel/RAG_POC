# pages/2_Upload_Chat.py
import streamlit as st
from common import (
    ensure_state, sidebar_status, render_messages,
    save_uploaded_files, _save_last_query,
    run_retrieval_test_upload, answer_query_upload, top_k_chunks,
    EXTERNAL_FILES_DIR
)
from common import render_upload_controls

def choose_retrieval_params(context_count: int) -> tuple[int, int]:
    """
    Gibt (per_doc_cap, topK0) basierend auf der Kontextanzahl zurück.

    Logik:
    - per_doc_cap = 3, außer bei mehr als 5 Dateien → dann 2
    - topK0 = mindestens 6, sonst 6 + (context_count * 5)

    Beispiel:
    - 2 Dateien → per_doc_cap=3, topK0=16
    - 6 Dateien → per_doc_cap=2, topK0=36
    """
    # Sicherheitscheck
    context_count = max(0, int(context_count))

    # per_doc_cap-Logik
    per_doc_cap = 3 if context_count <= 5 else 2

    # topK0-Logik
    topK0 = max(6, 6 + (context_count-1) * 5)

    return per_doc_cap, topK0


st.set_page_config(page_title="Upload-Chat", page_icon="📎", layout="wide")
ensure_state()
sidebar_status()

# Rechts-Spalte sticky machen (nur Desktop)
st.markdown("""
<style>
/* Gesamtbreite vergrößern */
.main .block-container { max-width: 1400px; }
/* Rechte Spalte sticky lassen */
@media (min-width: 768px){
  div[data-testid="column"]:nth-of-type(2) > div {
    position: sticky;
    top: 1rem;
    align-self: flex-start;
  }
}
/* Chips lesbarer machen */
code { white-space: pre-wrap; word-break: break-word; }
</style>
""", unsafe_allow_html=True)

st.title("📎 Upload-Chat")
st.caption("Dateien hochladen, Kontext wählen – Fragen laufen nur auf dem Kontext.")

# Zweispaltiges Layout
left_col, right_col = st.columns([3, 2], gap="large") 

# -------------------------
# RECHTS: Upload & Kontext (sticky)
# -------------------------
with right_col:
    st.markdown("### Dateien hochladen")
    up = st.file_uploader(
        "Dateien auswählen",
        type=["pdf","png","jpg","jpeg","tif","tiff","bmp","txt","md"],
        accept_multiple_files=True,
        help=None,  # Hilfetext spart Breite – alternativ in Expander
    )
    c1, c2 = st.columns([1,1])
    render_upload_controls()

    uploaded_paths = []
    if up:
        saved = save_uploaded_files(up)
        if saved:
            st.success(f"{len(saved)} Datei(en) gespeichert.")
            uploaded_paths = saved
            with st.expander("Gespeicherte Dateien"):
                for p in saved:
                    st.write(f"• {p.name}  \t({p.stat().st_size} Bytes)")

    st.divider()

    st.markdown("### 🧩 Kontext-Steuerung")
    available_files = sorted([p.name for p in EXTERNAL_FILES_DIR.iterdir() if p.is_file()])

    selection = st.multiselect(
        "Dateien auswählen, die dem Kontext hinzugefügt werden sollen:",
        options=available_files,
        default=[],
        placeholder="Wähle 1–10 Dateien aus",
        help="Nur UI-State. Die eigentliche Filter-Logik fügst du später an."
    )

    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Kontext hinzufügen", use_container_width=True, type="primary", disabled=(len(selection) == 0)):
            existing = set(st.session_state["context_doc_ids"])
            to_add   = [s for s in selection if s not in existing]
            st.session_state["context_doc_ids"].extend(to_add)
            st.success(f"{len(to_add)} Datei(en) zum Kontext hinzugefügt.")
    with cols[1]:
        if st.button("Kontext leeren", use_container_width=True):
            st.session_state["context_doc_ids"].clear()
            st.info("Kontext geleert.")

    st.markdown("#### Kontext:")
    if not st.session_state["context_doc_ids"]:
        st.caption("*(Kein Kontext gesetzt – spätere Abfragen laufen im Standardverhalten.)*")
    else:
        chip_cols = st.columns(3)
        for i, fname in enumerate(st.session_state["context_doc_ids"]):
            with chip_cols[i % 3]:
                st.code(fname, language=None)

    st.caption("Hinweis: Aktuell nur UI. Die Filter-Logik (z. B. `doc_id in context`) fügst du später an.")

# -------------------------
# LINKS: Chat-Verlauf & Verarbeitung
# -------------------------
with left_col:
    message_key = "messages_Upload-Chat"
    render_messages(message_key)

    user_input = st.chat_input("Frage eingeben und Enter drücken …")

    if user_input:
        st.session_state[message_key].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        _save_last_query(user_input)
        st.toast("Query gespeichert.", icon="✅")

        # 🚧 Guard: Kontext muss gesetzt sein
        ctx_files = st.session_state.get("context_doc_ids", [])
        per_doc_cap, topK0 = choose_retrieval_params(len(ctx_files))
        print("asdf1",per_doc_cap)
        print("asdf2",topK0)
        if not ctx_files:
            with st.chat_message("assistant"):
                st.info("Bitte wähle rechts unter **Kontext-Steuerung** mindestens eine Datei aus und stelle deine Frage erneut.")
            st.session_state[message_key].append({
                "role": "assistant",
                "content": "ℹ️ Bitte wähle rechts unter **Kontext-Steuerung** mindestens eine Datei aus und stelle deine Frage erneut."
            })
            st.stop()  # bricht die weitere Ausführung ab (kein Retrieval/LLM)


        with st.spinner("Suche relevante Textstellen (Top-K)…"):
            try:
                chunks = run_retrieval_test_upload(
                    user_input,
                    topK0=topK0,
                    per_doc_cap=per_doc_cap,
                    context_filenames=st.session_state.get("context_doc_ids", [])
                )
                
            except Exception as e:
                chunks = []
                st.error(f"Retrieval-Fehler: {e}")


        with st.spinner("Erzeuge Antwort …"):
            try:
                filtered_chunks = top_k_chunks(chunks, 30)
                print("asdf3",len(filtered_chunks))
                final_answer = answer_query_upload(user_input, filtered_chunks)
            except Exception as e:
                final_answer = f"_LLM-Fehler beim Erzeugen der Antwort:_ `{e}`"

        with st.chat_message("assistant"):
            st.markdown(final_answer)
        st.session_state[message_key].append({"role": "assistant", "content": final_answer})
