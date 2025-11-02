
from pathlib import Path
from typing import List, Tuple, Dict, Any
from typing import Optional
import re
import unicodedata
import time
from openai import AzureOpenAI
from collections import defaultdict
import json
from math import sqrt
from pathlib import Path
import asyncio
import traceback


from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient as AioDIClient

import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery, QueryType
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, VectorSearch,
    HnswAlgorithmConfiguration, VectorSearchProfile,
    SearchFieldDataType, VectorSearchAlgorithmKind,
    SearchField, LexicalAnalyzerName, SemanticSearch, 
    SemanticConfiguration, SemanticPrioritizedFields, SemanticField
)
from SectionBuilder import sections_from_paragraph_sign_chars
from dotenv import load_dotenv

#maximale parallele Anfragen an DocAI für OCR
MAX_CONCURRENCY=10

# .env laden
load_dotenv()

def require(name: str) -> str:
    """Hole eine Umgebungsvariable oder wirf einen klaren Fehler, falls sie fehlt."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


#nano 4.1 modell
AZURE_OPENAI_ENDPOINT_GPT4_1 = require("AZURE_OPENAI_ENDPOINT_GPT4_1")
AZURE_OPENAI_API_KEY_GPT4_1  = require("AZURE_OPENAI_API_KEY_GPT4_1")
GPT4_1_NANO_DEPLOYMENT       = require("GPT4_1_NANO_DEPLOYMENT")
GPT4_1_MODEL_VERSION         = require("GPT4_1_MODEL_VERSION")

#nano 5 modell
AZURE_OPENAI_ENDPOINT_GPT5 = require("AZURE_OPENAI_ENDPOINT_GPT5")
AZURE_OPENAI_API_KEY_GPT5  = require("AZURE_OPENAI_API_KEY_GPT5")
GPT5_NANO_DEPLOYMENT       = require("GPT5_NANO_DEPLOYMENT")
GPT5_MODEL_VERSION         = require("GPT5_MODEL_VERSION")

#AI SEARCH
endpoint  = require("AI_SEARCH_ENDPOINT")
admin_key = require("AI_SEARCH_ADMIN_KEY")

ix_client = SearchIndexClient(endpoint, AzureKeyCredential(admin_key))
doc_client = SearchClient(endpoint, "chunks", AzureKeyCredential(admin_key))

# --- neu: Volltext speichern ---
def save_full_text(full_text: str, out_dir: Path, base_name: str):
    """
    Speichert den gesamten Dokumenttext als UTF-8 in ./GesamtText/<basename>_FULL.txt
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_base = _safe_filename(base_name, max_len=80)
    out_path = out_dir / f"{safe_base}_FULL.txt"
    out_path.write_text(full_text, encoding="utf-8")
    print(f"[FULLTEXT] gespeichert: {out_path.resolve()}")


index = SearchIndex(
    name="chunks",
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="doc_id", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="chunk_no", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
        SearchableField(
            name="heading",
            type=SearchFieldDataType.String,
            analyzer_name=LexicalAnalyzerName.DE_MICROSOFT
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            analyzer_name=LexicalAnalyzerName.DE_MICROSOFT
        ),
        SearchField(
            name="meta",
            type=SearchFieldDataType.ComplexType,
            fields=[
                SimpleField(name="property_code", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
                SearchField(name="unit_codes", type=SearchFieldDataType.Collection(SearchFieldDataType.String), filterable=True, facetable=True),
                SimpleField(name="street", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="house_no", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="street_plus_house", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="postal_code", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
                SimpleField(name="city", type=SearchFieldDataType.String, filterable=True, facetable=True),
                SimpleField(name="tenant", type=SearchFieldDataType.String, filterable=True, facetable=True),
                # optional, falls du sie später nutzt:
                SimpleField(name="start_date", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
                SimpleField(name="fixed_term_end", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
            ]
        ),
        # Vektorfeld:
        SearchField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_profile_name="vprof"),
    ],
    vector_search=VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw", kind=VectorSearchAlgorithmKind.HNSW)],
        profiles=[VectorSearchProfile(name="vprof", algorithm_configuration_name="hnsw")]
    )
)

#index auf reranker ausrichten mit semantic suche
index.semantic_search = SemanticSearch(
    configurations=[
        SemanticConfiguration(
            name="default",
            prioritized_fields=SemanticPrioritizedFields(
                # "heading" als Titel/Überschrift
                title_field=SemanticField(field_name="heading"),
                # Hauptinhalt
                content_fields=[SemanticField(field_name="content")],
                # Zusatz-"Keywords" aus deinen Meta-Subfeldern (Pfadnotation)
                keywords_fields=[
                    SemanticField(field_name="meta/property_code"),
                    SemanticField(field_name="meta/street_plus_house"),
                    SemanticField(field_name="meta/street"),
                    SemanticField(field_name="meta/postal_code"),
                    SemanticField(field_name="meta/city"),
                    SemanticField(field_name="meta/tenant"),
                    SemanticField(field_name="meta/unit_codes"),
                ],
            )
        )
    ]
)

_allowed = re.compile(r"[^A-Za-z0-9_\-=]+")  # für IDs in Azure AI Search

def sanitize_for_search_key(s: str) -> str:
    # 1) Normalisieren, damit z.B. "a\u0308" sichtbar wird
    s = unicodedata.normalize("NFKD", s)

    # 2) Deutsche Umlaute explizit mappen (inkl. kombinierter Form)
    #    ([AaOoUu])\u0308 = Vokal + kombinierender Trema -> Vokal + e
    s = re.sub(r'([Aa])\u0308', lambda m: ('Ae' if m.group(1).isupper() else 'ae'), s)
    s = re.sub(r'([Oo])\u0308', lambda m: ('Oe' if m.group(1).isupper() else 'oe'), s)
    s = re.sub(r'([Uu])\u0308', lambda m: ('Ue' if m.group(1).isupper() else 'ue'), s)

    # Falls zusammengesetzte Zeichen vorkommen sollten, auch die mappen:
    s = s.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
    s = s.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')

    # ß -> ss
    s = s.replace('ß', 'ss')

    # 3) Übrige kombinierende Zeichen entfernen (z.B. übrig gebliebene Tilden etc.)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')

    # 4) Nur erlaubte Zeichen für Azure AI Search Keys
    s = _allowed.sub('-', s).strip('-')

    # 5) Mehrere Trenner zusammenziehen
    s = re.sub(r'[-_]{2,}', '-', s)

    return s







# =================== Konfiguration für DocAI===================
ENDPOINT = "https://docai-poc007.cognitiveservices.azure.com/"
API_KEY  = "8c2pAijYkNJPTpdd5mx8hPOGvgJeMeKYdpE9QnUlP2WWftWVmSUNJQQJ99BJACPV0roXJ3w3AAALACOGkrk0"

AZURE_OPENAI_ENDPOINT = "https://loren-mginqgpa-switzerlandnorth.cognitiveservices.azure.com/"
AZURE_OPENAI_API_KEY = "Bs8ew1C1DAE1GRTmJxeQ0QdML9PKewSV3CDwUhd8LvEVNmIok9stJQQJ99BJACI8hq2XJ3w3AAAAACOGgQ46"  
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"  # dein Deploymentname im Azure Studio
API_VERSION = "2024-02-01"


RETRIEVED_DIR = Path("./Retrieved")
RETRIEVED_DIR.mkdir(exist_ok=True)

INPUT_DIR  = Path("./Input")
OUTPUT_DIR = Path("./Result")
SLEEP_BETWEEN_FILES_SEC = 0.0


def get_document_initial_heading(full_text: str) -> str:
    """Erste nicht-leere Zeile als initiale Überschrift."""
    for ln in (full_text or "").splitlines():
        if ln.strip():
            return ln.strip()
    return "Abschnitt"

# ---- Hardcoded-Test ----
RUN_RETRIEVAL_TEST = True
QUERY_FILE = Path("./queries/last_query.txt")
if QUERY_FILE.exists():
    QUERY = QUERY_FILE.read_text(encoding="utf-8").strip()
else:
    QUERY = "Fallback-Query, falls keine Datei vorhanden ist"

K = 5  # k nächste Nachbarn

# ====== Laufzeit-Schalter ======
RUN_DOC_AI   = True          # True: OCR+Chunking ausführen; False: nur Embeddings erzeugen
EMBED_FROM   = "nofiles"       # "files"  -> aus ./Result/*.txt
                             # "sections" -> aus der sections-Variable (nur wenn RUN_DOC_AI=True)
# ===============================


ACCEPTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# Heuristik-Grenzen (zeichengenau)
MAX_CHARS_WITHOUT_PAR_SIGN = 500
LOOKAHEAD_CHARS            = 200
MIN_CHARS_PER_CHUNK        = 200
# =====================================================

DEBUG = True
def dbg(msg: str):
    if DEBUG:
        print(msg)

def _resolve_source_from_embedding_filename(emb_file_name: str) -> str:
    """
    Leitet den Ursprungs-Dateinamen aus ./Input ab.
    Beispiel: 'Vertrag_003_embedding.json' -> sucht 'Vertrag.*' in INPUT_DIR
    und gibt den exakten Dateinamen (mit Endung) zurück, falls gefunden.
    """
    m = re.match(r"^(.*)_(\d{3,4})_embedding\.json$", emb_file_name)
    base = m.group(1) if m else emb_file_name.replace("_embedding.json", "")
    # Suche im Input-Ordner nach gleicher 'stem'
    candidates = list(INPUT_DIR.glob("*"))
    for fp in candidates:
        if fp.is_file() and fp.stem == base:
            return fp.name  # exakter Ursprungs-Dateiname inkl. Endung
    # Fallback, falls nicht gefunden
    return base


def _slug(s: str, max_len: int = 60) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return (s[:max_len] or "query").strip("-")


def _embed_query(text: str) -> list[float]:
    """
    Holt ein Embedding für die Query (nutzt deine bestehende Azure OpenAI Config).
    """
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION
    )
    resp = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,  # <- dein Deployment-Name
        input=text
    )
    return resp.data[0].embedding

def _safe_filename(s: str, max_len: int = 80) -> str:
    # Unicode → ASCII
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # Alles Unsaubere durch '-' ersetzen (keine Slashs, keine Backslashes, keine Doppelpunkte etc.)
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    # führende/trailing Trenner weg
    s = s.strip("._-")
    # auf max Länge kürzen
    return s[:max_len] or "chunk"




def run_retrieval_test(
    query: str,
    *,
    use_hybrid: bool = True,
    topK0: int = 60,
    per_doc_cap: int = 3
):
    """
    Holt topK0 Kandidaten (Hybrid optional) und wendet NUR per_doc_cap an.
    Es werden alle Chunks behalten, solange die Cap pro doc_id nicht überschritten wird.
    """
    #print(f"[RET] Start | hybrid={use_hybrid} | topK0={topK0} | per_doc_cap={per_doc_cap}")

    # 1) Query-Embedding für Vektor-Teil
    try:
        q_emb = _embed_query(query)
    except Exception as e:
        print(f"[RET] Abbruch: Embedding-Fehler für Query: {e}")
        return

    vq = VectorizedQuery(vector=q_emb, k_nearest_neighbors=topK0, fields="embedding")

    # 2) Azure Search: Hybrid Suche und reranker mit semantic search
    try:
        if use_hybrid:
            results_iter = doc_client.search(
                search_text=query,                # BM25
                query_type=QueryType.SEMANTIC, 
                semantic_configuration_name="default",
                vector_queries=[vq],              # + Vector (Hybrid-Fusion)
                top=topK0
            )
        else:
            results_iter = doc_client.search(
                search_text=None,
                vector_queries=[vq],
                top=topK0
            )
    except Exception as e:
        print(f"[RET] Abbruch: AI Search Query-Fehler: {e}")
        return

    # 3) Materialisieren & nach Server-Score sortieren oder nach search.reranker_score
    results_list = list(results_iter)
    def _score(r):
        return r.get("@search.reranker_score", None)
    # fallback auf @search.score, wenn kein reranker_score vorhanden (z.B. ohne SEMANTIC)
    results = sorted(
        results_list,
        key=lambda r: (_score(r) is None, -float(r.get("@search.reranker_score", 0.0)) if _score(r) is not None else -float(r.get("@search.score", 0.0)))
    )

    # 4) per_doc_cap strikt anwenden (kein zweites Auffüllen)
    picked: List[Dict[str, Any]] = []
    per_doc: Dict[str, int] = {}

    for r in results:
        did = r.get("doc_id") or "(unknown)"
        cnt = per_doc.get(did, 0)
        if per_doc_cap is not None and cnt >= per_doc_cap:
            continue
        picked.append(r)
        per_doc[did] = cnt + 1

    print(f"[RET] per_doc Verteilung: {per_doc} | behalten={len(picked)} von server={len(results)}")

    # 5) Dateien schreiben
    qslug = _safe_filename(_slug(query), max_len=60)
    RETRIEVED_DIR.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        f"# Retrieval-Ergebnis (AI Search) für: {query}",
        f"- topK0 = {topK0}",
        f"- per_doc_cap = {per_doc_cap}",
        f"- behalten = {len(picked)}",
        ""
    ]
    returned_chunks: List[dict] = []

    for rank, r in enumerate(picked, start=1):
        heading = r.get("heading", "") or ""
        text    = r.get("content", "") or ""
        did     = r.get("doc_id", "") or ""
        score   = r.get("@search.reranker_score", r.get("@search.score"))
        meta    = r.get("meta",{})

        did_slug  = _safe_filename(did, max_len=60)
        head_slug = _safe_filename(heading, max_len=60)

        filename = f"{qslug}_cap_top{rank:02d}_{did_slug}_{head_slug}.txt"
        out_file = RETRIEVED_DIR / filename
        content  = (
            f"[RANK] {rank}\n"
            f"[AI SEARCH score] {score if score is not None else 'n/a'}\n"
            f"[SOURCE doc_id] {did}\n"
            f"[HEADING] {heading}\n\n"
            f"{text}\n"
        )
        out_file.write_text(content, encoding="utf-8")
        summary_lines.append(f"- **#{rank}** score={score if score is not None else 'n/a'} — `{out_file.name}`")

        # chunk_no robust bestimmen
        chunk_no_val = r.get("chunk_no")
        try:
            chunk_no = int(chunk_no_val) if chunk_no_val is not None else rank
        except Exception:
            chunk_no = rank

        returned_chunks.append({
            "doc_id": did,
            "chunk_no": chunk_no,
            "heading": heading,
            "content": text,
            "score": score,
            "meta" : meta,
        })

    # 6) Summary speichern (ohne k)
    summary_path = RETRIEVED_DIR / f"{qslug}_capped_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[RET] Übersicht gespeichert: {summary_path.name}")

    print("chunky", returned_chunks)
    return returned_chunks

def slugify(text: str, max_len: int = 60) -> str:
    import unicodedata, re
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return (text[:max_len] or "section").strip("-")


def first_non_empty_line(text: str) -> str:
    for ln in text.splitlines():
        if ln.strip():
            return ln.strip()
    return ""

#parallel OCR
async def _ocr_one_file(
    client: AioDIClient,
    fp: Path,
    sem: asyncio.Semaphore
) -> Tuple[Path, str]:
    """
    Eine Datei asynchron mit prebuilt-read verarbeiten, Ergebnis als (Path, Volltext).
    Fängt eigene Fehler ab, damit der Batch weiterlaufen kann.
    """
    async with sem:
        t0 = time.time()
        print(f"[START] {fp.name} → OCR startet …")
        try:
            # Datei-Bytes im Thread lesen (blockierendes IO nicht im Event-Loop)
            pdf_bytes = await asyncio.to_thread(fp.read_bytes)
            print(f"  [OK]  {fp.name}: {len(pdf_bytes):,} Bytes eingelesen")

            poller = await client.begin_analyze_document(
                model_id="prebuilt-read",
                body=pdf_bytes,
                output_content_format="text",
                pages=None,  # gesamtes Dokument
            )
            print(f"  [REQ] {fp.name}: Request an DI geschickt")

            result = await poller.result()
            text = (getattr(result, "content", "") or "")
            dt = time.time() - t0
            print(f"[DONE] {fp.name}: OCR fertig (len={len(text):,}, {dt:.1f}s)")
            return fp, text

        except Exception as e:
            dt = time.time() - t0
            print(f"❌ [FAIL] {fp.name}: {type(e).__name__}: {e} (nach {dt:.1f}s)")
            traceback.print_exc()
            return fp, ""


async def ocr_batch_async(files: List[Path], max_concurrency: int = MAX_CONCURRENCY) -> Dict[Path, str]:
    """
    Führt OCR für alle Dateien parallel aus und liefert {Path: Volltext}.
    Sorgt für gute Logs und robuste Fehlerbehandlung.
    """
    sem = asyncio.Semaphore(max_concurrency)
    out: Dict[Path, str] = {}

    print(f"\n[INFO] OCR-Batch gestartet: {len(files)} Dateien | max parallel = {max_concurrency}\n")

    try:
        # HIER: AIO-Client benutzen und via async context managen
        async with AioDIClient(endpoint=ENDPOINT, credential=AzureKeyCredential(API_KEY)) as client:
            print(f"[DBG] Client-Typ: {type(client)!r}  (sollte AioDIClient sein)")
            tasks = []
            for fp in files:
                if not fp.exists():
                    print(f"⚠️  Datei fehlt/ungültig: {fp}")
                    continue
                tasks.append(_ocr_one_file(client, fp, sem))

            print(f"[DBG] {len(tasks)} Tasks erzeugt → asyncio.gather() …\n")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            print("[DBG] gather() beendet – Auswertung …")

    except Exception as e:
        print(f"❌ [FATAL] Fehler beim Aufbau/await der Tasks: {type(e).__name__}: {e}")
        traceback.print_exc()
        return out  # leer zurück, damit Aufrufer nicht crasht

    for res in results:
        if isinstance(res, Exception):
            print(f"⚠️  Ausnahme-Objekt aus gather(): {res}")
            traceback.print_exc()
            continue
        fp, text = res
        out[fp] = text
        print(f"  [COLLECT] {fp.name}: Textlänge = {len(text):,}")

    print(f"\n[INFO] OCR-Batch fertig. Erfolgreich: {len(out)} / angefragt: {len(files)}\n")
    return out


def get_pdf_page_count(fp: Path) -> Optional[int]:
    if fp.suffix.lower() != ".pdf":
        return None
    try:
        from PyPDF2 import PdfReader  # optional
        total = len(PdfReader(str(fp)).pages)
        dbg(f"  [DBG] PyPDF2 erkannt: total_pages={total}")
        return total
    except Exception as e:
        dbg(f"  [DBG] PyPDF2 nicht verfügbar/Fehler: {e}")
        return None


def save_sections(sections: List[Tuple[str, str]], out_dir: Path, base_name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (heading, text) in enumerate(sections, start=1):
        name = f"{base_name}_{i:03d}_{slugify(heading)}.txt"
        (out_dir / name).write_text(text, encoding="utf-8")
    # kleine Vorschau
    previews = [h[:80] for h, _ in sections[:3]]
    dbg(f"  [DBG] gespeichert: {len(sections)} Dateien | Preview Headings: {previews}")


def load_sections_from_result_files(result_dir: Path) -> List[Tuple[str, str]]:
    """
    Liest alle gespeicherten .txt-Chunks aus dem Ordner ./Result
    und erzeugt eine Liste (heading, text).
    heading = erster Satz oder erste Zeile.
    """
    print(f"[EMB] Lade vorhandene Chunks aus: {result_dir.resolve()}")
    txt_files = sorted(result_dir.glob("*.txt"))
    sections = []
    for fp in txt_files:
        content = fp.read_text(encoding="utf-8").strip()
        if not content:
            continue
        heading = content.split("\n", 1)[0][:100]  # erste Zeile als Heading
        sections.append((heading, content))
    print(f"[EMB] {len(sections)} Chunk-Dateien geladen.")
    return sections



def extract_core_meta_llm_from_sections(
    sections: List[Tuple[str, str]],
    base_name: str,
    take_sections: int = 2,
) -> dict:
    """
    Extrahiert sichere Vertragsmetadaten per GPT-5-nano.
    Nutzt nur die ersten 1–2 Sections (Titel + Präambel).
    Felder:
      - property_code
      - unit_codes[]
      - street
      - house_no
      - street_plus_house
      - postal_code
      - city
      - tenant
    """
    # Nur Anfang des Textes verwenden (meist Vertragskopf)
    head_text = "\n\n".join(sec for _, sec in sections[:take_sections]).strip()
    head_text = head_text[:8000]

    client = AzureOpenAI(
        api_version=GPT4_1_MODEL_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_GPT4_1,
        api_key=AZURE_OPENAI_API_KEY_GPT4_1,
    )

    system = (
        "Du bist ein präziser Extraktionsassistent für Verträge. "
        "Lies den folgenden Vertragskopf und gib ausschließlich die geforderten Felder "
        "im JSON-Format zurück. Keine zusätzlichen Erklärungen, kein Fließtext.\n\n"

        "Definitionen der Felder:\n"
        "- property_code: interner Objektcode mit 3–4 Buchstaben + 2 Ziffern (z. B. 'ABC12', 'XYZ34', 'MUC56'). "
        "  Steht oft im Namen der Vermieterin/GmbH und ist NICHT Teil der Adresse.\n"

        "- unit_codes: Liste ALLER Einheitenkennungen im Text. Erkenne und extrahiere Varianten wie:\n"
        "  • 'WE 01.02', 'WE 03/12', 'WE 2.07'\n"
        "  • 'ME#0.02', 'ME#1.06', 'ME 0.02' (auch ohne #)\n"
        "  • Formatvarianten: optionales '#', Trennzeichen '.', '/', oder Leerzeichen\n"
        "  • Normalisiere so: Präfix GROSS (WE/ME), genau ein Leerzeichen nach Präfix, "
        "    wenn '#' vorhanden dann 'ME#0.02', sonst 'ME 0.02'; bei WE immer 'WE 02.07' (mit führenden Nullen)\n"
        "  • Duplizierte Codes entfernen, Reihenfolge des Auftretens beibehalten\n"
        "  • NICHT erfinden. Postleitzahlen, Hausnummern oder Flächenangaben sind KEINE unit_codes.\n"

        "- street: nur der Straßenname (z. B. 'Musterstraße'), ohne Hausnummer.\n"
        "- house_no: nur die Hausnummer (z. B. '12a').\n"
        "- street_plus_house: 'Straße + Hausnummer' (z. B. 'Musterstraße 12a').\n"
        "- postal_code: 5-stellige Zahl (z. B. '12345').\n"
        "- city: Stadtname (z. B. 'Beispielstadt').\n"
        "- start_date: Mietbeginn im Format YYYY-MM-DD, falls angegeben.\n"
        "- fixed_term_end: Ende der Festmietzeit im Format YYYY-MM-DD, falls angegeben.\n"
        "- tenant: Name der Mieterin (Firma/Person) ohne Zusatztexte.\n\n"

        "Wichtig für unit_codes:\n"
        "- Erkenne sowohl Wohn- ('WE') als auch Miet-/Gewerbeeinheiten ('ME').\n"
        "- Beispiele (nur zur Form, keine Werte übernehmen): "
        "  'WE 01.02', 'WE 03/12', 'ME#0.02', 'ME 1.06'.\n"
        "- Wenn eine Schreibweise wie 'WE03.12' ohne Leerzeichen vorkommt, normalisiere zu 'WE 03.12'.\n"

        "Wenn du etwas nicht sicher weißt, setze null oder []. Gib NUR gültiges JSON zurück.\n\n"

        "Beispielausgabe:\n"
        "{\n"
        '  \"property_code\": \"ABC12\",\n'
        '  \"unit_codes\": [\"WE 01.02\", \"ME#0.02\", \"WE 03/12\"],\n'
        '  \"street\": \"Musterstraße\",\n'
        '  \"house_no\": \"12a\",\n'
        '  \"street_plus_house\": \"Musterstraße 12a\",\n'
        '  \"postal_code\": \"12345\",\n'
        '  \"city\": \"Beispielstadt\",\n'
        '  \"start_date\": \"2024-11-01\",\n'
        '  \"fixed_term_end\": \"2034-10-31\",\n'
        '  \"tenant\": \"Supermarkt Beispiel GmbH\"\n'
        "}"
    )


    user = f"Kontext:\n{head_text}"

    try:
        resp = client.chat.completions.create(
            model=GPT4_1_NANO_DEPLOYMENT,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            max_completion_tokens=1000,
            temperature=0.0
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            meta = json.loads(raw)
        except Exception:
            m = re.search(r'\{.*\}', raw, re.S)
            meta = json.loads(m.group(0)) if m else {}
    except Exception as e:
        print(f"⚠️ Meta-Extraktionsfehler: {e}")
        meta = {}

    # Fallback + Normalisierung
    meta = {
        "property_code": meta.get("property_code") if isinstance(meta, dict) else None,
        "unit_codes": meta.get("unit_codes") if isinstance(meta, dict) else [],
        "street": meta.get("street") if isinstance(meta, dict) else None,
        "house_no": meta.get("house_no") if isinstance(meta, dict) else None,
        "street_plus_house": meta.get("street_plus_house") if isinstance(meta, dict) else None,
        "postal_code": meta.get("postal_code") if isinstance(meta, dict) else None,
        "city": meta.get("city") if isinstance(meta, dict) else None,
        "tenant": meta.get("tenant") if isinstance(meta, dict) else None,
        "start_date": meta.get("start_date") if isinstance(meta, dict) else None,
        "fixed_term_end": meta.get("fixed_term_end") if isinstance(meta, dict) else None,
    }

    if isinstance(meta["unit_codes"], str):
        meta["unit_codes"] = [u.strip() for u in meta["unit_codes"].split(",") if u.strip()]
    if not isinstance(meta["unit_codes"], list):
        meta["unit_codes"] = []

    # Falls street_plus_house nicht gesetzt wurde, automatisch kombinieren
    if not meta.get("street_plus_house") and (meta.get("street") or meta.get("house_no")):
        s = meta.get("street") or ""
        h = meta.get("house_no") or ""
        meta["street_plus_house"] = f"{s.strip()} {h.strip()}".strip() or None

    return meta




def create_and_save_embeddings(sections: List[Tuple[str, str]], base_name: str):
    aoai = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=API_VERSION
    )

    batch = []
    current_bytes = 0
    BYTES_LIMIT = 14_000_000      # konservativ < 16MB payload limit


    meta = extract_core_meta_llm_from_sections(sections, base_name)

    for i, (heading, text) in enumerate(sections, start=1):
        try:
            emb_resp = aoai.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=text)
            emb = emb_resp.data[0].embedding
        except Exception as e:
            print(f"  ❌ Embedding-Fehler Chunk {i}: {e}")
            continue

        base_name_saniticed = sanitize_for_search_key(base_name)
        doc = {
            "id": f"{base_name_saniticed}_{i:05d}",
            "doc_id": base_name,
            "chunk_no": i,
            "heading": heading,
            "content": text,
            "embedding": emb,
            "meta": meta,
        }
        # simple payload-size guard
        est_bytes = len(json.dumps(doc))
        if current_bytes + est_bytes > BYTES_LIMIT or len(batch) >= 500:
            try:
                doc_client.merge_or_upload_documents(batch)
                print(f"  ✅ {len(batch)} Chunks in AI Search geladen.")
            except Exception as e:
                print(f"  ❌ Upload-Fehler: {e}")
            batch, current_bytes = [], 0

        batch.append(doc)
        current_bytes += est_bytes

    if batch:
        try:
            doc_client.merge_or_upload_documents(batch)
            print(f"  ✅ {len(batch)} Chunks in AI Search geladen (Final).")
        except Exception as e:
            print(f"  ❌ Upload-Fehler (Final): {e}")


def main_embedding_mode(base_name: str, use_saved_files: bool = True, sections: Optional[List[Tuple[str, str]]] = None):
    """
    Steuert, ob die Embeddings aus vorhandenen Dateien oder aus der Variablen 'sections' erzeugt werden.
    """
    if use_saved_files:
        print("[EMB] Modus: Dateien -> Embeddings")
        sections = load_sections_from_result_files(OUTPUT_DIR)
    else:
        print("[EMB] Modus: Variable -> Embeddings")
        if sections is None:
            print("  ❌ Keine Sections übergeben!")
            return

    create_and_save_embeddings(sections, base_name)
    
from collections import defaultdict

def group_result_files_by_base(result_dir: Path) -> dict[str, list[Path]]:
    """
    Gruppiert ./Result/*.txt nach Basisname vor _NNN_ (z. B. 'Vertrag_001_...' -> 'Vertrag').
    So können wir pro Ursprungsdatei Embeddings erzeugen, ohne OCR erneut zu starten.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for fp in sorted(result_dir.glob("*.txt")):
        stem = fp.stem
        # Suche Trenner _NNN_ oder _NNNN_
        m = re.search(r"^(.*)_\d{3,4}_(?:.+)$", stem)
        base = m.group(1) if m else stem
        groups[base].append(fp)
    return groups

def load_sections_from_specific_files(files: list[Path]) -> List[Tuple[str, str]]:
    """
    Lädt (heading, text) nur aus den angegebenen Chunk-Dateien.
    """
    sections: List[Tuple[str, str]] = []
    for fp in files:
        content = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        heading = content.split("\n", 1)[0][:100]
        sections.append((heading, content))
    return sections


def top_k_chunks(chunks: List[Dict], k: int) -> List[Dict]:
    """
    Gibt die k besten Chunks nach 'score' zurück.

    Parameter:
        chunks: Liste von Dictionaries, z. B. [{"text": "...", "score": 0.87}, ...]
        k: Anzahl der Top-Ergebnisse

    Rückgabe:
        Liste der k besten Chunks, absteigend nach Score sortiert
    """
    return sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)[:k]


def filter_chunks_iterative_exact(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Iterativer Filter über Meta-Felder.
    - Reihenfolge der Felder ist maßgeblich.
    - Pro Feld: Substring-Check (case-insensitive) gegen die Query.
    - unit_codes ist eine Liste: Treffer, wenn irgendein Eintrag vorkommt.
    - Wenn ein Feld Treffer liefert, wird die Arbeitsmenge auf genau diese Treffer gesetzt.
    - Wenn kein Feld je Treffer liefert, werden die unveränderten 'chunks' zurückgegeben.
    """
    if not query or not chunks:
        return chunks

    fields = (
        "property_code",
        "unit_codes",
        "street",
        "street_plus_house",
        "postal_code",
        "city",
        "tenant",
    )
    print("länge halt: ",len(chunks))
    def norm(s: str) -> str:
        # einfache Normalisierung: trim, Mehrfachspaces -> 1 Space, lowercase
        return " ".join(str(s).split()).casefold()

    q_norm = norm(query)

    working_indices = list(range(len(chunks)))  # bewahrt Original-Reihenfolge
    filtered_indices = []  # wird nur gesetzt, wenn ein Feld Treffer liefert

    for field in fields:
        new_filtered = []

        for idx in working_indices:
            meta = chunks[idx].get("meta", {}) or {}
            val = meta.get(field)
            print("val: ",val)
            if not val:
                continue

            # Werte-Liste bauen
            if isinstance(val, list):
                values = [str(v) for v in val if v not in (None, "")]
            else:
                values = [str(val)]

            # Substring-Match gegen Query (genauer Stringvergleich, aber als Teilstring)
            hit = False
            for v in values:
                v_norm = norm(v)
                if v_norm and v_norm in q_norm:
                    hit = True
                    break

            if hit:
                print("treffer", field)
                new_filtered.append(idx)

        # Wenn dieses Feld Treffer hat, setzen wir auf die neue Menge um
        if new_filtered:
            filtered_indices = new_filtered
            working_indices = new_filtered
        # sonst: working_indices bleiben wie sie sind und wir versuchen das nächste Feld

        # Frühabbruch optional: wenn nichts mehr übrig ist
        if not working_indices:
            break

    # Wenn nie ein Feld Treffer hatte → alles zurückgeben
    if not filtered_indices:
        return chunks

    # Treffer-Chunks in Original-Reihenfolge zurückgeben
    print("länge halt: ",len(filtered_indices))
    return [chunks[i] for i in filtered_indices]


def format_context(chunks: List[dict]) -> str:
    lines = []
    for rank, r in enumerate(chunks, start=1):
        src = r.get("doc_id")
        head = (r.get("heading") or "").strip()
        pfrom, pto = r.get("page_from"), r.get("page_to")
        pages = f"pages {pfrom}-{pto}" if pfrom and pto else (f"page {pfrom}" if pfrom else "")
        lines.append(f"[{rank}] source={src} | {head} | {pages} ")
        text = (r.get("content") or "").strip()
        lines.append(f"\"\"\"\n{text}\n\"\"\"\n")
    return "\n".join(lines)

def answer_query(query: str, chunks: list[dict]) -> str:
    """
    Baut User Prompt mit Kontext, speichert ihn als ./userPrompt/prompt.txt,
    ruft GPT-5-nano über Azure auf und gibt die Antwort zurück.
    """
    ctx = format_context(chunks)

    system = (
        "Du bist ein präziser Assistent. Antworte ausschließlich anhand der Auszüge im CONTEXT. "
        "Belege jeden relevanten Satz mit einer Quelle im Format [#rank]. "
        "Wenn Informationen fehlen oder unklar sind, sage das explizit. Antworte auf Deutsch."
    )

    user = f"""
Frage: {query}

Lies die folgenden Textauszüge genau. Jeder Abschnitt ist nummeriert [#rank].
Verwende ausschließlich diese Inhalte, um zu antworten und beweise deine Aussagen anhand von Zitaten mit Angabe der Datei (source), der es entspringt.

CONTEXT:
{ctx}
"""

    # === 🔹 User Prompt speichern ===
    user_prompt_dir = Path("./userPrompt")
    user_prompt_dir.mkdir(exist_ok=True)
    prompt_path = user_prompt_dir / "prompt.txt"
    prompt_path.write_text(user.strip(), encoding="utf-8")
    print(f"[INFO] User Prompt gespeichert unter: {prompt_path.resolve()}")

    # === 🔹 Azure OpenAI Request ===
    client = AzureOpenAI(
        api_version=GPT5_MODEL_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT_GPT5,
        api_key=AZURE_OPENAI_API_KEY_GPT5,
    )

    try:
        resp = client.chat.completions.create(
            model=GPT5_NANO_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_completion_tokens=3000,
            extra_body={"reasoning_effort": "low"},
        )
        print("Antwortobjekt empfangen.")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM-Fehler beim Erzeugen der Antwort: {e}")
        return f"(Fehler bei der LLM-Abfrage: {e})"



def main():
    ok = 0
    fail = 0

    try:
        ix_client.create_index(index)
    except Exception:
        ix_client.delete_index("chunks")
        ix_client.create_index(index)

    # Vorbereitungen
    if not INPUT_DIR.exists():
        print(f"[FEHLT] Eingabeordner nicht gefunden: {INPUT_DIR.resolve()}")
        return

    # Liste der Eingabedateien einsammeln
    files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in ACCEPTED_EXTS])
    if not files:
        print(f"[LEER] Keine unterstützten Dateien in {INPUT_DIR.resolve()} gefunden.")
        return

    # ====== Embeddings-only Modus (RUN_DOC_AI=False) ======
    if not RUN_DOC_AI:
        print("[MODE] RUN_DOC_AI=False -> überspringe OCR/Chunking. Erzeuge NUR Embeddings aus ./Result/*.txt")
        if not OUTPUT_DIR.exists():
            print(f"[FEHLT] Result-Ordner {OUTPUT_DIR.resolve()} nicht gefunden.")
            return

        groups = group_result_files_by_base(OUTPUT_DIR)
        if not groups:
            print(f"[LEER] Keine Chunk-Dateien in {OUTPUT_DIR.resolve()} gefunden.")
            return

        for base, files_for_base in groups.items():
            print(f"[EMB-ONLY] '{base}': {len(files_for_base)} Chunk-Dateien")
            sections = load_sections_from_specific_files(files_for_base)
            create_and_save_embeddings(sections, base)
        print("[ENDE] Embeddings-only abgeschlossen.")
        return
    # ======================================================

    #Einmal alle OCR-Jobs parallel starten und Ergebnisse einsammeln
    texts_by_file = asyncio.run(ocr_batch_async(files, max_concurrency=MAX_CONCURRENCY))
    for idx, fp in enumerate(files, start=1):
        base_name = fp.stem
        print(f"[{idx}/{len(files)}] Datei: {fp.name} -> doc_id='{base_name}'")

        try:
            plain = (texts_by_file.get(fp) or "").strip()
            if not plain:
                print("  ❌ Kein Text extrahiert (oder Fehler im OCR-Batch).")
                fail += 1
                continue

            # Chunking (aus deiner SectionBuilder-Funktion)
            sections: List[Tuple[str, str]] = sections_from_paragraph_sign_chars(plain)
            if not sections:
                print("  ❌ Keine Sektionen erkannt.")
                fail += 1
                continue


            save_sections(sections, OUTPUT_DIR, base_name)
            print(f"  OK: {len(sections)} Sektionen gespeichert (z. B. {base_name}_001_<heading>.txt)")
            ok += 1

            # Embeddings je nach Quelle erzeugen
            if EMBED_FROM == "files":
                main_embedding_mode(base_name, use_saved_files=True,  sections=sections)
            else:
                main_embedding_mode(base_name, use_saved_files=False, sections=sections)

        except HttpResponseError as e:
            print(f"  ERROR (Azure): {e}")
            fail += 1
        except Exception as e:
            print(f"  ERROR (Allg.): {e}")
            fail += 1

        if SLEEP_BETWEEN_FILES_SEC > 0:
            time.sleep(SLEEP_BETWEEN_FILES_SEC)

    print(f"[ENDE] Erfolgreich: {ok} | Fehlgeschlagen: {fail}")


if __name__ == "__main__":
    main()
