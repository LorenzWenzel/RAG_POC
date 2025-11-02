import re
from typing import List, Tuple, Optional
from pathlib import Path
import unicodedata
from typing import Union

SIMPLE_HEADING_LINE_RE = re.compile(r"(?m)^[ \t]{0,3}§\s*\d+\b.*$")
# ----------- Tokenizer / Hilfsfunktionen -----------
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENT_BOUNDARY_RE = re.compile(r"([.!?…;:]+)(?=\s*[\)\]»›\"'„“‚’]*\s*[A-ZÄÖÜ§0-9])")

def extract_simple_heading(line: str, max_len: int = 100) -> Optional[str]:
    """
    Nimmt die komplette Zeile als Heading, wenn sie mit '§' + Zahl beginnt und <= max_len ist.
    Sonst None.
    """
    if not line:
        return None
    if SIMPLE_HEADING_LINE_RE.match(line):
        h = " ".join(line.strip().split())   # Whitespace glätten
        if len(h) <= max_len:
            return h
    return None

#nur aus testzwecken: 
def _safe_filename(s: str, max_len: int = 80) -> str:
    """Robuster, kurzer Dateiname ohne Sonderzeichen/Whitespaces."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("._-")
    if not s:
        s = "chunk"
    return s[:max_len]

def write_test_result(
    sections: List[Tuple[str, str]],
    out_dir: Union[Path, str] = Path("./TESTERGEBNIS"),
    base_name: str = "doc",
) -> Path:

    """
    Schreibt die (heading, text)-Tupel als einzelne .txt-Dateien nach ./TESTERGEBNIS.
    Zusätzlich wird eine _index.tsv mit Übersicht erzeugt.
    Rückgabe: Pfad zum Ausgabeverzeichnis.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    base_safe = _safe_filename(base_name, max_len=60)
    index_lines = ["#\tfilename\theading\tchars"]

    for i, (heading, content) in enumerate(sections, start=1):
        head_safe = _safe_filename(heading or "Abschnitt", max_len=60)
        fname = f"{base_safe}_{i:03d}_{head_safe}.txt"
        fpath = out_path / fname
        fpath.write_text(content or "", encoding="utf-8")

        index_lines.append(f"{i}\t{fname}\t{(heading or '').strip()}\t{len(content or '')}")

    # kleine Index-Datei (TSV) zur schnellen Übersicht
    (out_path / f"{base_safe}__index.tsv").write_text(
        "\n".join(index_lines), encoding="utf-8"
    )

    print(f"[TESTERGEBNIS] {len(sections)} Dateien in: {out_path.resolve()}")
    return out_path

def tokenize_with_spans(text: str):
    toks, spans = [], []
    for m in _TOKEN_RE.finditer(text):
        toks.append(m.group(0)); spans.append((m.start(), m.end()))
    return toks, spans, len(text)

def token_index_from_char(spans, char_pos: int) -> int:
    lo, hi = 0, len(spans)
    while lo < hi:
        mid = (lo + hi) // 2
        if spans[mid][0] < char_pos: lo = mid + 1
        else: hi = mid
    return lo

def detok_slice(text: str, spans, i_from: int, i_to: int) -> str:
    if i_from >= i_to: return ""
    start = spans[i_from][0]; end = spans[i_to-1][1]
    return text[start:end]

def _line_break_token_positions(text: str, spans) -> List[int]:
    pos = []
    for m in re.finditer(r"\n+", text):
        idx = token_index_from_char(spans, m.end())  # erstes Token nach dem/n \n
        pos.append(idx)
    return pos

def _heading_line_token_positions(text: str, spans) -> List[int]:
    pos = []
    for m in SIMPLE_HEADING_LINE_RE.finditer(text):
        pos.append(token_index_from_char(spans, m.start()))
    return pos

def _sentence_boundary_token_positions(text: str, spans) -> List[int]:
    pos = []
    for m in SENT_BOUNDARY_RE.finditer(text):
        pos.append(token_index_from_char(spans, m.end()))
    return pos

# ----------- Chunker (unverändert schlank, aber mit Snap auf §-Heading-Zeilen) -----------
def chunk_tokenwise_with_line_snap(
    text: str,
    TARGET: int = 800,
    OVERLAP: int = 100,
    MIN: int = 600,
    MAX: int = 1000,
    GRACE: int = 150,
    LINE_SNAP: int = 80,
    SENT_SNAP: int = 80,
    LINE_BACK: int = 120,
    SENT_BACK: int = 120,
    # Start-Snap nach dem Overlap-Backjump:
    START_LINE_FWD: int = 60,    # vorwärts zum nächsten Zeilenanfang
    START_SENT_FWD: int = 60,    # vorwärts zur nächsten Satzgrenze
    START_LINE_BACK: int = 200,  # rückwärts zum letzten Zeilenanfang
    START_SENT_BACK: int = 200,  # rückwärts zur letzten Satzgrenze
) -> List[str]:
    """
    Chunkt `text` tokenbasiert mit 'sanftem' Vorwärts-Snap des Chunkendes
    (Zeilenanfang / Satzende) und *zusätzlichem* Start-Snap des nächsten Chunks
    NACH dem Overlap-Backjump, damit Chunks nicht mitten im Satz/Zeilenrest beginnen.

    Rückgabe: Liste von reinen Chunk-Strings (ohne Headings).
    """
    toks, spans, _ = tokenize_with_spans(text)
    n = len(toks)
    if n == 0:
        return []

    # Vorindexierte Ankerpunkte
    line_starts = _line_break_token_positions(text, spans)     # Token-Index NACH '\n'
    head_pos    = _heading_line_token_positions(text, spans)   # §+Zahl am Zeilenanfang
    sent_pos    = _sentence_boundary_token_positions(text, spans)  # Token-Index NACH Satzende

    out: List[str] = []
    i = 0
    while i < n:
        end = min(i + TARGET, n)

        # (1) Falls kurz danach ein Heading kommt, dort schneiden
        if GRACE > 0:
            candidates = [p for p in head_pos if end <= p <= min(end + GRACE, n)]
            if candidates:
                end = candidates[0]

        chosen_forward = False

        # (2) Vorwärts zum nächsten Zeilenanfang (innerhalb LINE_SNAP)
        if end < n and LINE_SNAP > 0:
            lb = [p for p in line_starts if end <= p <= min(end + LINE_SNAP, n)]
            if lb:
                end = lb[0]
                chosen_forward = True

        # (3) Vorwärts zur nächsten Satzgrenze (innerhalb SENT_SNAP)
        if not chosen_forward and end < n and SENT_SNAP > 0:
            sb = [p for p in sent_pos if end <= p <= min(end + SENT_SNAP, n)]
            if sb:
                end = sb[0]
                chosen_forward = True

        # (4) Wenn vorwärts nichts Gutes, rückwärts schnappen (Zeile/Satz)
        if not chosen_forward:
            if LINE_BACK > 0:
                lb_back = [p for p in line_starts
                          if max(i + MIN, i) <= p <= end and p >= max(i, end - LINE_BACK)]
                if lb_back:
                    end = lb_back[-1]
            if SENT_BACK > 0 and end - i >= MIN:
                sb_back = [p for p in sent_pos
                          if max(i + MIN, i) <= p <= end and p >= max(i, end - SENT_BACK)]
                if sb_back:
                    end = sb_back[-1]

        # (5) MIN/MAX hart einhalten
        end = min(end, i + MAX, n)
        if end - i < MIN and i + MIN <= n:
            end = min(i + MIN, i + MAX, n)

        # Slice erzeugen
        out.append(detok_slice(text, spans, i, end))

        if end >= n:
            break

        # ========= Start-SNAP für den nächsten Chunk =========
        base = max(0, end - OVERLAP)

        # (A) Vorwärts zum nächsten Zeilenanfang ab 'base'
        fwd_line = [p for p in line_starts if base <= p <= min(base + START_LINE_FWD, n)]
        if fwd_line:
            i = fwd_line[0]
            continue

        # (B) Vorwärts zur nächsten Satzgrenze ab 'base'
        fwd_sent = [p for p in sent_pos if base <= p <= min(base + START_SENT_FWD, n)]
        if fwd_sent:
            i = fwd_sent[0]
            continue

        # (C) Rückwärts zum letzten Zeilenanfang vor 'base'
        back_line = [p for p in line_starts if max(0, base - START_LINE_BACK) <= p < base]
        if back_line:
            i = back_line[-1]
            continue

        # (D) Rückwärts zur letzten Satzgrenze vor 'base'
        back_sent = [p for p in sent_pos if max(0, base - START_SENT_BACK) <= p < base]
        if back_sent:
            i = back_sent[-1]
            continue

        # (E) Fallback: roher base-Index
        i = base
        # =====================================================
    # winzigen Rest ggf. anhängen
    if len(out) >= 2 and len(out[-1].split()) < (MIN // 2):
        out[-2] = out[-2] + ("\n\n" if out[-2] else "") + out[-1]
        out.pop()

    return out

# ----------- Hauptfunktion: Chunks + einfache Headings (aus §-Zeilen ≤ 50 Zeichen) -----------
def sections_from_paragraph_sign_chars(plain_text: str) -> List[Tuple[str, str]]:
    t = (plain_text or "").strip()
    if not t:
        return []

    chunks = chunk_tokenwise_with_line_snap(
        t, TARGET=800, OVERLAP=100, MIN=600, MAX=1000,
        GRACE=150, LINE_SNAP=80, SENT_SNAP=80, LINE_BACK=120, SENT_BACK=120
    )

    final: List[Tuple[str, str]] = []
    carry = chunks[0].strip().splitlines()[0] if chunks and chunks[0].strip() else "Mietvertrag"
    
    for chunk in chunks:
        heads: List[str] = [carry] if carry else []
        for line in chunk.splitlines():
            h = extract_simple_heading(line, max_len=100)
            if h and (not heads or h != heads[-1]) and h not in heads:
                heads.append(h)

        carry = heads[-1] if heads else carry
        final.append((", ".join(heads), chunk))
        print(heads)
    return final