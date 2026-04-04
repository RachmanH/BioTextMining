import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "data.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    conn = get_conn()
    c = conn.cursor()

    # Dokumen yang diupload
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )

    # Relasi mentah per kalimat (evidence level)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS relations_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            chemical TEXT NOT NULL,
            disease TEXT NOT NULL,
            sentence TEXT,
            confidence REAL NOT NULL DEFAULT 0.0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )

    # Entity mentah per kalimat untuk replay explainable (index-based highlight)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS entities_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            sentence TEXT NOT NULL,
            start_idx INTEGER NOT NULL,
            end_idx INTEGER NOT NULL,
            label TEXT NOT NULL,
            text TEXT NOT NULL,
            score REAL NOT NULL DEFAULT 0.0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )

    # Relasi agregat per dokumen untuk graph cepat
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS relations_agg (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            chemical TEXT NOT NULL,
            disease TEXT NOT NULL,
            count INTEGER NOT NULL DEFAULT 0,
            avg_conf REAL NOT NULL DEFAULT 0.0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
        )
        """
    )

    # Index penting untuk performa query
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_relations_raw_doc_id
        ON relations_raw(doc_id)
        """
    )

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_relations_raw_pair
        ON relations_raw(doc_id, chemical, disease)
        """
    )

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_relations_agg_doc_id
        ON relations_agg(doc_id)
        """
    )

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_entities_raw_doc_id
        ON entities_raw(doc_id)
        """
    )

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_entities_raw_sentence
        ON entities_raw(doc_id, sentence)
        """
    )

    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_relations_agg_pair
        ON relations_agg(doc_id, chemical, disease)
        """
    )

    conn.commit()
    conn.close()