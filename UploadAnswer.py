
from pathlib import Path
from typing import List, Tuple, Dict, Any
from typing import Optional
import re
import unicodedata
import time
import asyncio
from openai import AzureOpenAI
from collections import defaultdict
import json
from math import sqrt
from pathlib import Path

from azure.core.exceptions import HttpResponseError
from azure.ai.documentintelligence.aio import DocumentIntelligenceClient as AioDIClient
from azure.core.credentials import AzureKeyCredential

import os
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
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Eigene Chunking-Funktion importieren (wie bei dir)
# Erwartet: sections_from_paragraph_sign_chars(text) -> List[Tuple[heading, content]]
# ─────────────────────────────────────────────────────────────────────────────
from SectionBuilder import sections_from_paragraph_sign_chars

# =================== Konfiguration: SERVICES & PFADE ===================

# .env laden
load_dotenv()

def require(name: str) -> str:
    """Hole eine Umgebungsvariable oder wirf einen klaren Fehler, falls sie fehlt."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

# =================== Azure OpenAI GPT 5 nano ===================
AZURE_OPENAI_ENDPOINT_GPT5 = require("AZURE_OPENAI_ENDPOINT_GPT5")
AZURE_OPENAI_API_KEY_GPT5  = require("AZURE_OPENAI_API_KEY_GPT5")
GPT5_NANO_DEPLOYMENT       = require("GPT5_NANO_DEPLOYMENT")
GPT5_MODEL_VERSION         = require("GPT5_MODEL_VERSION")

# =================== Azure OpenAI Embeddings small ===================
AZURE_OPENAI_ENDPOINT = require("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY  = require("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT  = require("EMBEDDING_DEPLOYMENT")
EMBED_API_VERSION     = require("EMBED_API_VERSION")

# =================== Azure AI Search ==================
SEARCH_ENDPOINT     = require("AI_SEARCH_ENDPOINT")
SEARCH_ADMIN_KEY    = require("AI_SEARCH_ADMIN_KEY")
SEARCH_INDEX_NAME   = "uploadchunks"   # bleibt konstant im Code

# =================== Azure Document Intelligence ===================
DI_ENDPOINT = require("DI_ENDPOINT")
DI_API_KEY  = require("DI_API_KEY")


# INPUT: jetzt aus ExternalFiles
INPUT_DIR  = Path("./ExternalFiles")
INPUT_DIR.mkdir(exist_ok=True)

# Sonstige Settings
ACCEPTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".txt", ".md"}
DEBUG = True

MAX_CONCURRENCY = 10  # gleichzeitige OCR-Jobs


def dbg(msg: str):
    if DEBUG:
        print(msg)

def get_index_client() -> SearchIndexClient:
    return SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))

def get_search_client() -> SearchClient:
    # Immer den aktuellen Indexnamen verwenden
    return SearchClient(SEARCH_ENDPOINT, SEARCH_INDEX_NAME, AzureKeyCredential(SEARCH_ADMIN_KEY))


# =================== Utilities ===================
_allowed = re.compile(r"[^A-Za-z0-9_\-=]+")  # für IDs in Azure AI Search




def sanitize_for_search_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'([Aa])\u0308', lambda m: ('Ae' if m.group(1).isupper() else 'ae'), s)
    s = re.sub(r'([Oo])\u0308', lambda m: ('Oe' if m.group(1).isupper() else 'oe'), s)
    s = re.sub(r'([Uu])\u0308', lambda m: ('Ue' if m.group(1).isupper() else 'ue'), s)
    s = s.replace('Ä', 'Ae').replace('Ö', 'Oe').replace('Ü', 'Ue')
    s = s.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
    s = s.replace('ß', 'ss')
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    s = _allowed.sub('-', s).strip('-')
    s = re.sub(r'[-_]{2,}', '-', s)
    return s

def _slug(s: str, max_len: int = 60) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return (s[:max_len] or "query").strip("-")

def _safe_filename(s: str, max_len: int = 80) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s)
    s = s.strip("._-")
    return s[:max_len] or "chunk"

def _embed_query(text: str) -> List[float]:
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=EMBED_API_VERSION
    )
    resp = client.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=text)
    return resp.data[0].embedding


# =================== Index sicherstellen (Deutsch-Analyzer, Vector) ===================
def ensure_index_exists():
    ix_client = get_index_client()

    index = SearchIndex(
        name=SEARCH_INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),

            # Falls du doc_id semantisch (als Keyword) berücksichtigen willst,
            # MUSS es SearchableField sein:
            SearchableField(name="doc_id", type=SearchFieldDataType.String,
                            filterable=True, facetable=True),

            SimpleField(name="chunk_no", type=SearchFieldDataType.Int32,
                        filterable=True, sortable=True),

            SearchableField(name="heading", type=SearchFieldDataType.String,
                            analyzer_name=LexicalAnalyzerName.DE_MICROSOFT),

            SearchableField(name="content", type=SearchFieldDataType.String,
                            analyzer_name=LexicalAnalyzerName.DE_MICROSOFT),

            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=1536,
                vector_search_profile_name="vprof",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw", kind=VectorSearchAlgorithmKind.HNSW)],
            profiles=[VectorSearchProfile(name="vprof", algorithm_configuration_name="hnsw")]
        ),
    )

    # 🔎 Semantic-Ranker: nur heading + content (optional doc_id als Keywords)
    index.semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name="default",
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(field_name="heading"),
                    content_fields=[SemanticField(field_name="content")],
                    keywords_fields=[SemanticField(field_name="doc_id")],  # optional; weglassen, wenn nicht gewünscht
                ),
            )
        ]
    )

    ix_client.create_or_update_index(index)


# =================== OCR + Chunking + Embedding + Upload ===================
def analyze_text_range(client: AioDIClient, file_path: Path, pages: Optional[str]):
    with open(file_path, "rb") as f:
        dbg(f"[DI] analyze -> {file_path.name} | pages={pages}")
        poller = client.begin_analyze_document(
            model_id="prebuilt-read",
            body=f,
            output_content_format="text",
            pages=pages
        )
    return poller.result()



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
        async with AioDIClient(endpoint=DI_ENDPOINT, credential=AzureKeyCredential(DI_API_KEY)) as client:
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






def index_external_files_upload():
    print("hier hats zumindest geklappt")
    """
    Lädt ALLE Dateien aus ./ExternalFiles,
    führt OCR/Text-Extraktion durch, chunked den Text,
    erstellt Embeddings und lädt die Dokumente direkt in den AI Search Index.
    Es wird NICHTS auf die Platte geschrieben außer Logs/Prompts.
    """
    ensure_index_exists()
    doc_client = get_search_client()

    if not INPUT_DIR.exists():
        print(f"[FEHLT] INPUT_DIR fehlt: {INPUT_DIR.resolve()}")
        return

    files = sorted([p for p in INPUT_DIR.iterdir() if p.is_file() and p.suffix.lower() in ACCEPTED_EXTS])
    if not files:
        print(f"[LEER] Keine unterstützten Dateien in {INPUT_DIR.resolve()} gefunden.")
        return

    di_client = AioDIClient(endpoint=DI_ENDPOINT, credential=AzureKeyCredential(DI_API_KEY))
    aoai = AzureOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=EMBED_API_VERSION)

    #Einmal alle OCR-Jobs parallel starten und Ergebnisse einsammeln
    texts_by_file = asyncio.run(ocr_batch_async(files, max_concurrency=MAX_CONCURRENCY))
    ok, fail = 0, 0
    for idx, fp in enumerate(files, start=1):
        base = fp.stem
        base_name = sanitize_for_search_key(fp.stem)
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

            # Embeddings erzeugen und direkt in den Index hochladen
            batch: List[dict] = []
            current_bytes = 0
            BYTES_LIMIT = 14_000_000  # konservativ < 16MB payload limit

            for i, (heading, text) in enumerate(sections, start=1):
                try:
                    emb_resp = aoai.embeddings.create(model=EMBEDDING_DEPLOYMENT, input=text)
                    emb = emb_resp.data[0].embedding
                except Exception as e:
                    print(f"  ❌ Embedding-Fehler Chunk {i}: {e}")
                    continue

                doc = {
                    "id": f"{base_name}_{i:05d}",
                    "doc_id": base,
                    "chunk_no": i,
                    "heading": heading,
                    "content": text,
                    "embedding": emb,
                }

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

            ok += 1

        except HttpResponseError as e:
            print(f"  ERROR (Azure): {e}")
            fail += 1
        except Exception as e:
            print(f"  ERROR (Allg.): {e}")
            fail += 1

    print(f"[ENDE] Indexierung ExternalFiles -> Erfolgreich: {ok} | Fehlgeschlagen: {fail}")

def uploadFilesToIndex():
    # 0) Index-Setup
    ensure_index_exists()
    doc_client = get_search_client()
    # 1) (Re-)Indexierung nur, wenn ./ExternalFiles Dateien enthält
    if any(INPUT_DIR.glob("*")):
        try:
            print("[RET-UP] ExternalFiles nicht leer – starte (Re-)Indexierung.")
            index_external_files_upload()
        except Exception as e:
            print(f"[RET-UP] Warn: Indexierung schlug fehl oder war leer: {e}")
    else:
        print("[RET-UP] ExternalFiles leer – überspringe Indexierung.")
        

def delete_all_entries_in_index():
    """
    Löscht den gesamten AI Search Index und legt ihn neu an.
    Dadurch werden alle Inhalte entfernt, das Schema aber wiederhergestellt.

    Schneller und robuster als das Löschen einzelner Dokumente.
    """
    try:
        # Admin-Client holen (nicht den normalen SearchClient!)
        admin_client = get_search_admin_client()
    except Exception as e:
        print(f"[FEHLER] Konnte SearchAdminClient nicht initialisieren: {e}")
        return

    try:
        print(f"[INFO] Lösche gesamten Index '{SEARCH_INDEX_NAME}' ...")
        admin_client.delete_index(SEARCH_INDEX_NAME)
        print(f"🗑️ Index '{SEARCH_INDEX_NAME}' wurde erfolgreich gelöscht.")
    except Exception as e:
        print(f"[WARNUNG] Index konnte evtl. nicht gelöscht werden (nicht vorhanden?): {e}")

    try:
        print(f"[INFO] Erstelle Index '{SEARCH_INDEX_NAME}' neu ...")
        ensure_index_exists()
        print(f"✅ Index '{SEARCH_INDEX_NAME}' wurde erfolgreich neu erstellt (leer).")
    except Exception as e:
        print(f"[FEHLER] Neuerstellung des Index fehlgeschlagen: {e}")


# =================== Retrieval (Upload-Variante) ===================
def run_retrieval_test_upload(
    query: str,
    *,
    use_hybrid: bool = True,
    topK0: int = 60,
    per_doc_cap: int = 3,
    context_filenames: Optional[List[str]] = None
) -> List[dict]:
    print("hier schonmal da", context_filenames)
    """
    Wendet NUR per_doc_cap an (strikt).
    Gibt die behaltenen Chunks (dicts) zurück.
    """
    print(f"[RET-UP] Start | hybrid={use_hybrid} | topK0={topK0} | per_doc_cap={per_doc_cap}")

    # 0) Index-Setup
    doc_client = get_search_client()
    
    # 0.1) Optionaler Kontext-Filter
    filter_str = None
    if context_filenames:
        # Dateinamen bereinigen (Endung entfernen, Leerzeichen trimmen)
        unique_filenames = []
        for f in context_filenames:
            if not f or not f.strip():
                continue
            # Endung entfernen mit Path.stem → robust für beliebige Formate
            name_no_ext = Path(f.strip()).stem
            unique_filenames.append(name_no_ext)

        # Duplikate entfernen
        unique_filenames = list(set(unique_filenames))

        if not unique_filenames:
            print("[RET-UP] Kein Kontext angegeben → Rückgabe leer.")
            return []

        joined = ",".join(unique_filenames)
        filter_str = f"search.in(doc_id, '{joined}', ',')"
        print(f"[RET-UP] Kontextfilter aktiv: {filter_str}")

    try:
        q_emb = _embed_query(query)
    except Exception as e:
        print(f"[RET-UP] Abbruch: Embedding-Fehler für Query: {e}")
        return []

    vq = VectorizedQuery(vector=q_emb, k_nearest_neighbors=topK0, fields="embedding")

    try:
        if use_hybrid:
            results_iter = doc_client.search(
                search_text=query,                 # BM25
                vector_queries=[vq],               # + Vector (Hybrid-Fusion)
                query_type=QueryType.SEMANTIC, 
                semantic_configuration_name="default",
                top=topK0,
                filter=filter_str
            )
        else:
            results_iter = doc_client.search(
                search_text=None,
                vector_queries=[vq],
                top=topK0
            )
    except Exception as e:
        print(f"[RET-UP] Abbruch: AI Search Query-Fehler: {e}")
        return []

    # Materialisieren & nach Server-Score sortieren
    results = list(results_iter)
    results.sort(key=lambda r: (r.get("@search.reranker_score") is None,
                            -float(r.get("@search.reranker_score", 0.0)
                                if r.get("@search.reranker_score") is not None
                                else r.get("@search.score", 0.0))))
    print("adajlkvncjdth",len(results))
    # per_doc_cap strikt
    picked: List[Dict[str, Any]] = []
    per_doc: Dict[str, int] = {}
    for r in results:
        did = r.get("doc_id") or "(unknown)"
        cnt = per_doc.get(did, 0)
        if per_doc_cap is not None and cnt >= per_doc_cap:
            continue
        picked.append(r)
        per_doc[did] = cnt + 1

    print(f"[RET-UP] per_doc Verteilung: {per_doc} | behalten={len(picked)} von server={len(results)}")

    # In flache dicts transformieren
    returned_chunks: List[dict] = []
    for rank, r in enumerate(picked, start=1):
        heading = r.get("heading", "") or ""
        text    = r.get("content", "") or ""
        did     = r.get("doc_id", "") or ""
        score   = r.get("@search.reranker_score", None)

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
        })
    print("vtidfjgne",len(returned_chunks))
    return returned_chunks


# =================== Answer (Upload-Variante) ===================
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

def estimate_gpt5_nano_cost(resp) -> float:
    """
    Berechnet die Gesamtkosten einer GPT-5-nano-Antwort auf Basis der Token-Nutzung.

    Erwartet: Das vollständige Antwortobjekt `resp`, wie es von
    client.chat.completions.create(...) zurückgegeben wird.

    Gibt die Gesamtkosten in USD zurück.
    """
    print(resp)
    try:
        usage = resp.usage
        input_toks = usage.prompt_tokens
        completion_toks = usage.completion_tokens
        reasoning_toks = getattr(resp.usage.completion_tokens_details, "reasoning_tokens", 0)


        # === GPT-5-nano Pricing (Stand: 2025, "low reasoning effort") ===
        # Quelle: Azure OpenAI / OpenAI API (approximate)
        # Preise pro 1.000 Tokens:
        PRICE_INPUT = 0.05/1000      # z. B. $0.05 / 1M input tokens
        PRICE_OUTPUT = 0.4/1000    # z. B. $0.15 / 1M output tokens
        PRICE_REASONING = 0.4/1000   # z. B. $0.30 / 1M reasoning tokens

        cost_input = input_toks / 1000 * PRICE_INPUT
        cost_output = completion_toks / 1000 * PRICE_OUTPUT
        cost_reasoning = reasoning_toks / 1000 * PRICE_REASONING

        total = cost_input + cost_output + cost_reasoning

        print(f"--- GPT-5-nano Tokenkosten ---")
        print(f"Input-Tokens:     {input_toks:,} → ${cost_input:.6f}")
        print(f"Completion-Tokens:{completion_toks:,} → ${cost_output:.6f}")
        print(f"Reasoning-Tokens: {reasoning_toks:,} → ${cost_reasoning:.6f}")
        print(f"Gesamtkosten:     ${total:.6f}")
        print(f"Preis auf 1000 prompts gesehen: ${total*1000}")

        return total

    except Exception as e:
        print(f"[WARN] Kostenberechnung fehlgeschlagen: {e}")
        return 0.0


def answer_query_upload(query: str, chunks: List[dict]) -> str:
    """
    Baut User Prompt mit Kontext (aus Upload-Indexierung),
    ruft GPT-5-nano (Azure) und gibt die Antwort zurück.
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
        estimate_gpt5_nano_cost(resp)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM-Fehler beim Erzeugen der Antwort: {e}")
        return f"(Fehler bei der LLM-Abfrage: {e})"

