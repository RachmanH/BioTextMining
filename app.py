from __future__ import annotations

import uuid
import os
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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


def _insert_document_and_relations(
    doc_id: str,
    filename: str,
    relations: List[Dict[str, Any]],
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


@app.post("/ner/pdf", response_model=List[NerReResponse])
async def ner_pdf(files: List[UploadFile] = File(...)):
    responses: List[NerReResponse] = []

    for file in files:
        if file.content_type not in ("application/pdf",):
            continue

        data = await file.read()
        if not data:
            continue

        sentences = extract_pdf_sentences(data)
        author_list = build_author_list(sentences)

        all_entities: List[Dict[str, Any]] = []
        all_relations: List[Dict[str, Any]] = []

        for sent in sentences:
            entities = extract_entities(sent, author_list)
            all_entities.extend(entities)

            rels = extract_relations(sent, entities)
            all_relations.extend(rels)

        uniq_entities = _deduplicate_entities(all_entities)
        uniq_relations = deduplicate_relations(all_relations)

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
            relations = run_pipeline(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gagal menjalankan pipeline ({file.filename}): {e}")

        try:
            _insert_document_and_relations(doc_id, file.filename or "unknown.pdf", relations)
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