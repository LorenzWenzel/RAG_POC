from typing import List, Dict, Any

def apply_per_doc_cap(chunks: List[Dict[str, Any]], per_doc_cap: int = 3) -> List[Dict[str, Any]]:
    """
    Nimmt eine Liste von Chunks (mit 'doc_id' und '@search.score' bzw. 'score')
    und behält nur die besten Chunks nach Score,
    wobei pro Dokument maximal 'per_doc_cap' Chunks erlaubt sind.

    Parameter:
        chunks: Liste von Dicts mit Schlüsseln ['doc_id', 'chunk_no', 'heading', 'content', 'score']
        per_doc_cap: maximale Anzahl Chunks pro Dokument (z. B. 3 oder 4)

    Rückgabe:
        Gefilterte Liste sortiert nach Score absteigend.
    """

    # 1) Sortiere alle Chunks nach Score (absteigend)
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True)

    # 2) Zähle pro Dokument, lösche Überschuss
    per_doc_counter = {}
    filtered = []
    for ch in sorted_chunks:
        did = ch.get("doc_id")
        if not did:
            continue
        per_doc_counter[did] = per_doc_counter.get(did, 0)
        if per_doc_counter[did] < per_doc_cap:
            filtered.append(ch)
            per_doc_counter[did] += 1
        # alle weiteren Chunks desselben Dokuments werden ignoriert

    print(f"[INFO] {len(filtered)}/{len(chunks)} Chunks behalten "
          f"({len(chunks)-len(filtered)} entfernt, cap={per_doc_cap})")
    return filtered
