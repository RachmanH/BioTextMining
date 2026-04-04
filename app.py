from __future__ import annotations

from collections import defaultdict
import uuid
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aggregator import aggregate_relations
from db import get_conn, init_db
from pipeline import (
    build_author_list,
    deduplicate_relations,
    extract_entities,
    extract_pdf_sentences,
    extract_relations,
    run_pipeline,
    run_pipeline_with_entities,
)


app = FastAPI(title="PharmaViz Local API", version="1.0.0")
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Untuk development lokal; nanti production batasi origin frontend Anda.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class NerEntity(BaseModel):
    start: int
    end: int
    label: str
    text: str
    score: float


class RelationOut(BaseModel):
    sentence: str
    chemical: str
    disease: str
    rel_type: str
    chemical_conf: float
    disease_conf: float
    confidence: float


class NerReResponse(BaseModel):
    doc_id: str
    filename: str
    entities: List[NerEntity]
    relations: List[RelationOut]


DRUG_LABELS = {"CHEMICAL", "DRUG"}
EFFECT_LABELS = {"DISEASE", "ADR", "ADVERSE_EVENT"}
REPLAY_MAX_STEPS = 3
LIVE_STEPS_MAX_PER_FILE = 0


def _entity_key(e: Dict[str, Any]) -> Tuple[str, str, int, int, str]:
    return (
        str(e.get("sentence", "")).strip(),
        str(e.get("label", "")).upper(),
        int(e.get("start", -1)),
        int(e.get("end", -1)),
        str(e.get("text", "")).strip().lower(),
    )


def _dedupe_entities_with_sentence(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for e in entities:
        key = _entity_key(e)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(
            {
                "sentence": str(e.get("sentence", "")).strip(),
                "start": int(e.get("start", -1)),
                "end": int(e.get("end", -1)),
                "label": str(e.get("label", "")),
                "text": str(e.get("text", "")),
                "score": float(e.get("score", 0.0)),
            }
        )
    return uniq


def _insert_document_and_relations(
    doc_id: str,
    filename: str,
    relations: List[Dict[str, Any]],
    entities: List[Dict[str, Any]] | None = None,
) -> None:
    conn = get_conn()
    c = conn.cursor()

    try:
        # Simpan metadata dokumen
        c.execute(
            "INSERT INTO documents (id, filename) VALUES (?, ?)",
            (doc_id, filename),
        )

        # Simpan relasi mentah
        for r in relations:
            c.execute(
                """
                INSERT INTO relations_raw (doc_id, chemical, disease, sentence, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    r.get("chemical", "").strip(),
                    r.get("disease", "").strip(),
                    r.get("sentence", "").strip(),
                    float(r.get("confidence", 0.0)),
                ),
            )

        # Simpan entity mentah untuk replay explainable.
        for e in entities or []:
            sent = str(e.get("sentence") or "").strip()
            text = str(e.get("text") or "").strip()
            if not sent or not text:
                continue

            start_idx = int(e.get("start", -1))
            end_idx = int(e.get("end", -1))
            if start_idx < 0 or end_idx <= start_idx:
                continue

            c.execute(
                """
                INSERT INTO entities_raw (doc_id, sentence, start_idx, end_idx, label, text, score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    sent,
                    start_idx,
                    end_idx,
                    str(e.get("label", "")),
                    text,
                    float(e.get("score", 0.0)),
                ),
            )

        # Agregasi per dokumen
        agg = aggregate_relations(relations)

        # Simpan relasi agregat
        for a in agg:
            c.execute(
                """
                INSERT INTO relations_agg (doc_id, chemical, disease, count, avg_conf)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    a["chemical"],
                    a["disease"],
                    int(a["count"]),
                    float(a["avg_conf"]),
                ),
            )

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _deduplicate_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for e in entities:
        key = (
            int(e.get("start", -1)),
            int(e.get("end", -1)),
            str(e.get("label", "")).upper(),
            str(e.get("text", "")).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(
            {
                "start": int(e.get("start", -1)),
                "end": int(e.get("end", -1)),
                "label": str(e.get("label", "")),
                "text": str(e.get("text", "")),
                "score": float(e.get("score", 0.0)),
            }
        )
    return uniq


def _score_relation_for_replay(rel: Dict[str, Any]) -> float:
    return float(rel.get("confidence", 0.0))


def _build_fallback_span(sentence: str, text: str, label: str, score: float) -> Dict[str, Any]:
    sent_low = sentence.lower()
    needle = text.lower().strip()
    idx = sent_low.find(needle)
    if idx < 0:
        idx = 0
    end_idx = min(len(sentence), idx + len(text))
    return {
        "start": idx,
        "end": end_idx,
        "label": label,
        "text": text,
        "score": score,
    }


def _select_step_entities(
    sentence: str,
    chem: str,
    dis: str,
    entities_in_sentence: List[Dict[str, Any]],
    rel_conf: float,
) -> List[Dict[str, Any]]:
    chem_low = chem.lower()
    dis_low = dis.lower()

    chem_entities = [
        e
        for e in entities_in_sentence
        if str(e.get("label", "")).upper() in DRUG_LABELS
        and str(e.get("text", "")).strip().lower() == chem_low
    ]
    dis_entities = [
        e
        for e in entities_in_sentence
        if str(e.get("label", "")).upper() in EFFECT_LABELS
        and str(e.get("text", "")).strip().lower() == dis_low
    ]

    selected: List[Dict[str, Any]] = []
    if chem_entities:
        selected.append(max(chem_entities, key=lambda x: float(x.get("score", 0.0))))
    else:
        selected.append(_build_fallback_span(sentence, chem, "CHEMICAL", rel_conf))

    if dis_entities:
        selected.append(max(dis_entities, key=lambda x: float(x.get("score", 0.0))))
    else:
        selected.append(_build_fallback_span(sentence, dis, "DISEASE", rel_conf))

    selected.sort(key=lambda x: int(x.get("start", -1)))
    return selected


def _build_replay_payload(doc_id: str, max_steps: int = REPLAY_MAX_STEPS) -> Dict[str, Any]:
    conn = get_conn()
    c = conn.cursor()

    rel_rows = c.execute(
        """
        SELECT chemical, disease, sentence, confidence
        FROM relations_raw
        WHERE doc_id = ?
        ORDER BY confidence DESC
        """,
        (doc_id,),
    ).fetchall()

    ent_rows = c.execute(
        """
        SELECT sentence, start_idx, end_idx, label, text, score
        FROM entities_raw
        WHERE doc_id = ?
        ORDER BY sentence, start_idx
        """,
        (doc_id,),
    ).fetchall()

    conn.close()

    if not rel_rows:
        raise HTTPException(status_code=404, detail="Replay data tidak ditemukan")

    sentence_entities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sent, start_idx, end_idx, label, text, score in ent_rows:
        sentence_entities[str(sent)].append(
            {
                "start": int(start_idx),
                "end": int(end_idx),
                "label": str(label),
                "text": str(text),
                "score": float(score),
            }
        )

    for sent in sentence_entities:
        sentence_entities[sent].sort(key=lambda x: int(x.get("start", -1)))

    rel_dicts: List[Dict[str, Any]] = [
        {
            "chemical": str(chem),
            "disease": str(dis),
            "sentence": str(sent),
            "confidence": float(conf),
        }
        for chem, dis, sent, conf in rel_rows
    ]

    rel_dicts.sort(key=_score_relation_for_replay, reverse=True)

    used = set()
    steps: List[Dict[str, Any]] = []
    for rel in rel_dicts:
        chem = str(rel["chemical"]).strip()
        dis = str(rel["disease"]).strip()
        sent = str(rel["sentence"]).strip()
        conf = float(rel["confidence"])
        if not chem or not dis or not sent:
            continue

        dedupe_key = (chem.lower(), dis.lower(), sent.lower())
        if dedupe_key in used:
            continue
        used.add(dedupe_key)

        entities = _select_step_entities(
            sentence=sent,
            chem=chem,
            dis=dis,
            entities_in_sentence=sentence_entities.get(sent, []),
            rel_conf=conf,
        )

        steps.append(
            {
                "sentence": sent,
                "entities": entities,
                "relation": {
                    "chemical": chem,
                    "disease": dis,
                    "confidence": conf,
                },
            }
        )

        if len(steps) >= max_steps:
            break

    if not steps:
        raise HTTPException(status_code=404, detail="Replay steps tidak tersedia")

    return {
        "doc_id": doc_id,
        "steps": steps,
        "timing": {
            "modal_open_ms": 200,
            "entity_each_ms": 280,
            "relation_ms": 380,
            "evidence_ms": 380,
            "modal_close_ms": 180,
        },
    }


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _truncate_sentence(sentence: str, max_len: int = 180) -> str:
    s = str(sentence or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


@app.post("/ner/pdf", response_model=List[NerReResponse])
async def ner_pdf(files: List[UploadFile] = File(...)):
    responses: List[NerReResponse] = []

    for file in files:
        if file.content_type not in ("application/pdf",):
            continue

        data = await file.read()
        if not data:
            continue

        result = run_pipeline_with_entities(data)
        uniq_entities = _deduplicate_entities(result["entities"])
        uniq_relations = deduplicate_relations(result["relations"])

        responses.append(
            NerReResponse(
                doc_id=str(uuid.uuid4()),
                filename=file.filename or "unknown.pdf",
                entities=[NerEntity(**e) for e in uniq_entities],
                relations=[RelationOut(**r) for r in uniq_relations],
            )
        )

    if not responses:
        raise HTTPException(status_code=400, detail="No valid PDF files found or processed.")

    return responses


@app.post("/ner/pdf/simple")
async def ner_pdf_simple(files: List[UploadFile] = File(...)):
    try:
        results = []

        for file in files:
            if file.content_type not in ("application/pdf",):
                continue

            data = await file.read()
            if not data:
                continue

            sentences = extract_pdf_sentences(data)
            author_list = build_author_list(sentences)

            chemicals = set()
            diseases = set()

            for sent in sentences:
                entities = extract_entities(sent, author_list)
                for e in entities:
                    label = str(e.get("label", "")).upper()
                    text = str(e.get("text", "")).strip()
                    if not text:
                        continue
                    if label in {"CHEMICAL", "DRUG"}:
                        chemicals.add(text)
                    if label in {"DISEASE", "ADR", "ADVERSE_EVENT"}:
                        diseases.add(text)

            results.append(
                {
                    "filename": file.filename,
                    "chemicals": sorted(list(chemicals)),
                    "diseases": sorted(list(diseases)),
                }
            )

        if not results:
            raise HTTPException(status_code=400, detail="No valid PDF files found or processed.")

        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_pdf(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Tidak ada file yang diupload")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maksimal upload 5 file PDF")

    documents = []

    for file in files:
        if file.content_type not in {"application/pdf"}:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' bukan PDF")

        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' kosong")

        doc_id = str(uuid.uuid4())

        try:
            extraction = run_pipeline_with_entities(data)
            relations = extraction["relations"]
            entities = _dedupe_entities_with_sentence(extraction["entities"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal menjalankan pipeline ({file.filename}): {e}")

        try:
            _insert_document_and_relations(
                doc_id,
                file.filename or "unknown.pdf",
                relations,
                entities,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal menyimpan ke database ({file.filename}): {e}")

        documents.append(
            {
                "doc_id": doc_id,
                "filename": file.filename,
                "relations_count": len(relations),
            }
        )

    # Compatibility fields for single-file clients.
    first = documents[0]
    return {
        "status": "ok",
        "count": len(documents),
        "documents": documents,
        "doc_id": first["doc_id"],
        "filename": first["filename"],
        "relations_count": first["relations_count"],
    }


@app.post("/upload/stream")
async def upload_pdf_stream(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Tidak ada file yang diupload")

    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maksimal upload 5 file PDF")

    loaded_files: List[Tuple[str, bytes]] = []
    for f in files:
        if f.content_type not in {"application/pdf"}:
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' bukan PDF")

        data = await f.read()
        if not data:
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' kosong")
        loaded_files.append((f.filename or "unknown.pdf", data))

    def stream_events():
        documents: List[Dict[str, Any]] = []
        total_files = len(loaded_files)

        try:
            yield _sse_event("status", {"message": "Memulai pemrosesan dokumen"})

            for file_idx, (filename, data) in enumerate(loaded_files, start=1):
                doc_id = str(uuid.uuid4())
                yield _sse_event(
                    "file_start",
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "file_index": file_idx,
                        "file_total": total_files,
                    },
                )

                try:
                    sentences = extract_pdf_sentences(data)
                    author_list = build_author_list(sentences)
                except Exception as e:
                    yield _sse_event(
                        "error",
                        {"message": f"Gagal ekstraksi PDF ({filename}): {str(e)}"},
                    )
                    return

                all_relations: List[Dict[str, Any]] = []
                all_entities: List[Dict[str, Any]] = []
                emitted_steps = 0
                total_sentences = max(1, len(sentences))

                for sent_idx, sent in enumerate(sentences, start=1):
                    try:
                        entities = extract_entities(sent, author_list=author_list)
                        rels = extract_relations(sent, entities)
                    except Exception as e:
                        yield _sse_event(
                            "error",
                            {"message": f"Gagal inferensi kalimat ({filename}): {str(e)}"},
                        )
                        return

                    all_relations.extend(rels)
                    for e in entities:
                        all_entities.append(
                            {
                                "sentence": sent,
                                "start": int(e.get("start", -1)),
                                "end": int(e.get("end", -1)),
                                "label": str(e.get("label", "")),
                                "text": str(e.get("text", "")),
                                "score": float(e.get("score", 0.0)),
                            }
                        )

                    progress = sent_idx / total_sentences
                    yield _sse_event(
                        "progress",
                        {
                            "doc_id": doc_id,
                            "filename": filename,
                            "progress": progress,
                            "sentence_index": sent_idx,
                            "sentence_total": total_sentences,
                        },
                    )

                    if not rels:
                        continue

                    for rel in rels:
                        if LIVE_STEPS_MAX_PER_FILE > 0 and emitted_steps >= LIVE_STEPS_MAX_PER_FILE:
                            break

                        selected_entities = _select_step_entities(
                            sentence=sent,
                            chem=str(rel.get("chemical", "")),
                            dis=str(rel.get("disease", "")),
                            entities_in_sentence=[
                                {
                                    "start": int(en.get("start", -1)),
                                    "end": int(en.get("end", -1)),
                                    "label": str(en.get("label", "")),
                                    "text": str(en.get("text", "")),
                                    "score": float(en.get("score", 0.0)),
                                }
                                for en in entities
                            ],
                            rel_conf=float(rel.get("confidence", 0.0)),
                        )

                        yield _sse_event(
                            "step",
                            {
                                "doc_id": doc_id,
                                "filename": filename,
                                "sentence": _truncate_sentence(sent, 180),
                                "entities": selected_entities,
                                "relation": {
                                    "chemical": str(rel.get("chemical", "")),
                                    "disease": str(rel.get("disease", "")),
                                    "confidence": float(rel.get("confidence", 0.0)),
                                },
                            },
                        )
                        emitted_steps += 1

                relations = deduplicate_relations(all_relations)
                entities = _dedupe_entities_with_sentence(all_entities)

                try:
                    _insert_document_and_relations(doc_id, filename, relations, entities)
                except Exception as e:
                    yield _sse_event(
                        "error",
                        {"message": f"Gagal menyimpan ke database ({filename}): {str(e)}"},
                    )
                    return

                documents.append(
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "relations_count": len(relations),
                    }
                )

                yield _sse_event(
                    "file_done",
                    {
                        "doc_id": doc_id,
                        "filename": filename,
                        "relations_count": len(relations),
                    },
                )

            first = documents[0]
            yield _sse_event(
                "complete",
                {
                    "status": "ok",
                    "count": len(documents),
                    "documents": documents,
                    "doc_id": first["doc_id"],
                    "filename": first["filename"],
                    "relations_count": first["relations_count"],
                },
            )
        except Exception as e:
            yield _sse_event("error", {"message": f"Internal stream error: {str(e)}"})

    return StreamingResponse(stream_events(), media_type="text/event-stream")


@app.get("/graph/{doc_id}")
def get_graph(doc_id: str):
    conn = get_conn()
    c = conn.cursor()

    rows = c.execute(
        """
        SELECT chemical, disease, count, avg_conf
        FROM relations_agg
        WHERE doc_id = ?
        """,
        (doc_id,),
    ).fetchall()

    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="doc_id tidak ditemukan atau belum ada hasil")

    nodes: Dict[str, Dict[str, str]] = {}
    links: List[Dict[str, Any]] = []

    for chem, dis, count, conf in rows:
        chem_key = chem.strip()
        dis_key = dis.strip()

        nodes[chem_key] = {"id": chem_key, "type": "drug"}
        nodes[dis_key] = {"id": dis_key, "type": "effect"}

        links.append(
            {
                "source": chem_key,
                "target": dis_key,
                "value": int(count),
                "confidence": float(conf),
            }
        )

    return {
        "doc_id": doc_id,
        "nodes": list(nodes.values()),
        "links": links,
    }


@app.get("/relations/{doc_id}")
def get_relations(doc_id: str):
    """
    Endpoint opsional untuk kebutuhan info panel/evidence sentence di frontend.
    """
    conn = get_conn()
    c = conn.cursor()

    rows = c.execute(
        """
        SELECT rr.doc_id, d.filename, rr.chemical, rr.disease, rr.sentence, rr.confidence
        FROM relations_raw rr
        LEFT JOIN documents d ON d.id = rr.doc_id
        WHERE rr.doc_id = ?
        ORDER BY rr.confidence DESC
        """,
        (doc_id,),
    ).fetchall()

    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Relasi mentah tidak ditemukan")

    return [
        {
            "doc_id": rel_doc_id,
            "filename": filename,
            "chemical": chem,
            "disease": dis,
            "sentence": sent,
            "confidence": float(conf),
        }
        for rel_doc_id, filename, chem, dis, sent, conf in rows
    ]


@app.get("/replay/{doc_id}")
def get_replay(doc_id: str):
    return _build_replay_payload(doc_id)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html tidak ditemukan")
    return FileResponse(index_path)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)