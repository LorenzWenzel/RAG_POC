# pages/1_RAG_Chat.py
import streamlit as st
from common import (
    ensure_state, sidebar_status, render_messages,
    run_retrieval_test, answer_query, filter_chunks_iterative_exact, top_k_chunks,
    _save_last_query, FIRST_K, PER_DOC_CAP, TOP_K
)

st.set_page_config(page_title="RAG-Chat", page_icon="🧠", layout="centered")
ensure_state()
sidebar_status()

st.title("🧠 RAG-Chat")
st.caption("Frage eingeben → Retrieval läuft → Top-K Treffer werden angezeigt → Antwort wird mit Quellen erstellt.")

# Verlauf
message_key = "messages_RAG-Chat"
render_messages(message_key)

# Eingabe
user_input = st.chat_input("Frage eingeben und Enter drücken …")

# Verarbeitung
if user_input:
    st.session_state[message_key].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    _save_last_query(user_input)
    st.toast("Query gespeichert.", icon="✅")

    with st.spinner("Suche relevante Textstellen (Top-K)…"):
        try:
            chunks = run_retrieval_test(user_input, topK0=FIRST_K, per_doc_cap=PER_DOC_CAP)
        except Exception as e:
            chunks = []
            st.error(f"Retrieval-Fehler: {e}")

    with st.spinner("Erzeuge Antwort …"):
        try:
            filtered_chunks = filter_chunks_iterative_exact(user_input, chunks)
            filtered_chunks = top_k_chunks(filtered_chunks, TOP_K)

            # with st.chat_message("assistant"):
            #     st.markdown(f"### 🔎 Gefilterte Treffer ({len(filtered_chunks)})")
            #     if not filtered_chunks:
            #         st.caption("*(Keine Filter-Treffer – verwende alle Chunks)*")
            #     to_show = filtered_chunks or chunks
            #     for i, r in enumerate(to_show, start=1):
            #         st.markdown(f"**Treffer {i}** — {r.get('heading','')}")
            #         st.caption(f"{r.get('doc_id')} · Chunk {int(r.get('chunk_no',0))}")
            #         preview = (r.get("content") or "")
            #         st.code(preview[:10000] + ("…" if len(preview) > 10000 else ""), language=None)
            #         meta = r.get("meta", {})
            #         if meta:
            #             st.markdown("**Metadaten:**")
            #             st.json(meta)
            #         else:
            #             st.caption("*(Keine Metadaten vorhanden)*")

            final_answer = answer_query(user_input, filtered_chunks or chunks)
        except Exception as e:
            final_answer = f"_LLM-Fehler beim Erzeugen der Antwort:_ `{e}`"

    with st.chat_message("assistant"):
        st.markdown(final_answer)
    st.session_state[message_key].append({"role": "assistant", "content": final_answer})
