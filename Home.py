# Home.py
import streamlit as st
from common import ensure_state, sidebar_status

st.set_page_config(page_title="RAG Query Chat (PoC)", page_icon="💬", layout="centered")
ensure_state()
st.title("💬 RAG Query Chat (PoC)")

st.markdown("""
Wähle oben links eine Seite:
- **RAG-Chat**: Suche im bestehenden Index (alles wie gehabt)
- **Upload-Chat**: Dateien hochladen, Kontext wählen, nur darauf fragen
""")

sidebar_status()

with st.expander("🔧 Diagnose", expanded=False):
    from common import HISTORY_FILE
    st.write(f"History-Datei: `{HISTORY_FILE.resolve()}`")
