from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aggregator import aggregate_relations
from db import get_conn, init_db
from pipeline import run_pipeline


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


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type not in {"application/pdf"}:
        raise HTTPException(status_code=400, detail="File harus PDF")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File kosong")

    doc_id = str(uuid.uuid4())

    try:
        relations = run_pipeline(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menjalankan pipeline: {e}")

    try:
        _insert_document_and_relations(doc_id, file.filename or "unknown.pdf", relations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan ke database: {e}")

    return {
        "doc_id": doc_id,
        "filename": file.filename,
        "relations_count": len(relations),
        "status": "ok",
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
        SELECT chemical, disease, sentence, confidence
        FROM relations_raw
        WHERE doc_id = ?
        ORDER BY confidence DESC
        """,
        (doc_id,),
    ).fetchall()

    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Relasi mentah tidak ditemukan")

    return [
        {
            "chemical": chem,
            "disease": dis,
            "sentence": sent,
            "confidence": float(conf),
        }
        for chem, dis, sent, conf in rows
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

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)