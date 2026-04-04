from __future__ import annotations

import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import nltk
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline as hf_pipeline


# =========================
# Config
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "albert_ner"

DEVICE = 0 if torch.cuda.is_available() else -1
SCORE_THRESH = float(os.getenv("SCORE_THRESH", "0.90"))

CAUSE_PATTERNS = [
    r"cause(s|d)?",
    r"induce(s|d)?",
    r"lead(s|ing)? to",
    r"triggers?",
    r"is associated with",
    r"results? in",
    r"contributes? to",
]

DRUG_LABELS = {"CHEMICAL", "DRUG"}
ADR_LABELS = {"DISEASE", "ADR", "ADVERSE_EVENT"}

JOURNAL_KEYWORDS = [
    "journal",
    "society",
    "association",
    "committee",
    "guideline",
    "task force",
    "european",
    "american",
    "soc cardiology",
    "heart association",
    "conference",
    "publisher",
    "clinical practice",
    "clinical oncology",
    "press",
    "university",
    "group",
]

GENERIC_WORDS = {
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "references",
    "table",
    "figure",
    "copyright",
    "preprint",
    "medrxiv",
    "biorxiv",
}


# =========================
# NLTK bootstrap
# =========================
def _ensure_nltk_resources() -> None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # Beberapa environment butuh punkt_tab
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


_ensure_nltk_resources()


# =========================
# Model bootstrap
# =========================
_NLP = None


def _load_hf_pipeline():
    global _NLP
    if _NLP is not None:
        return _NLP

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory tidak ditemukan: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_DIR))

    _NLP = hf_pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=DEVICE,
    )
    return _NLP


# =========================
# Filter helpers
# =========================
def filter_doi(entity: str) -> bool:
    entity = entity.strip().lower()
    patterns = [
        r"^doi[\s:\-\.]",
        r"^10\.\d{4,9}/",
        r"^[\w\-.]+@[\w\-.]+$",
        r"^\d+(\.\d+)+$",
        r"\bdoi\b",
        r"(preprint|medrxiv|biorxiv|arxiv)",
        r"^e\d{4,6}$",
    ]
    for pat in patterns:
        if re.search(pat, entity):
            return True

    if len(entity) < 5 and not entity.isalpha():
        return True

    if "doi.org" in entity or "http" in entity:
        return True

    return False


def filter_author(entity: str, author_list: Optional[List[str]] = None) -> bool:
    entity_clean = entity.lower().replace("-", " ").replace(",", " ").strip()
    entity_clean = re.sub(r"\s+", " ", entity_clean)

    if author_list:
        for a in author_list:
            a_clean = str(a or "").lower().strip()
            if not a_clean:
                continue
            # Exact/near-exact match to extracted author names.
            if (
                entity_clean == a_clean
                or entity_clean.startswith(a_clean + " ")
                or entity_clean.endswith(" " + a_clean)
            ):
                return True

    parts = entity.split()
    if len(parts) == 2 and len(parts[0]) == 1:
        return True

    if re.match(r"^[a-z]{1,2}\s[a-z]{2,}$", entity_clean):
        return True

    return False


def filter_journal(entity: str) -> bool:
    entity = entity.strip().lower()
    return any(k in entity for k in JOURNAL_KEYWORDS)


def filter_year(entity: str) -> bool:
    entity = entity.strip()
    return bool(re.match(r"^(19|20|21)\d{2}$", entity))


def filter_generic(entity: str) -> bool:
    return entity.strip().lower() in GENERIC_WORDS


def should_filter_entity(entity: str, author_list: Optional[List[str]] = None) -> bool:
    if not entity or len(entity.strip()) <= 2:
        return True
    if filter_doi(entity):
        return True
    if filter_author(entity, author_list):
        return True
    if filter_journal(entity):
        return True
    if filter_year(entity):
        return True
    if filter_generic(entity):
        return True
    return False


# =========================
# Core pipeline functions
# =========================
def clean_entity_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("##", "")
    text = text.replace("▁", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,.;:()[]{}")
    return text.strip()


def extract_pdf_sentences(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []
    for page in doc:
        all_text.append(page.get_text("text"))
    raw_text = "\n".join(all_text)
    raw_text = re.sub(r"\s+", " ", raw_text).strip()

    if not raw_text:
        return []

    sentences = nltk.sent_tokenize(raw_text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def build_author_list(sentences: List[str], take_n: int = 5) -> List[str]:
    author_list: List[str] = []
    for s in sentences[:take_n]:
        parts = re.split(r",|and", s)
        for p in parts:
            token = p.strip()
            if 2 < len(token) < 30 and token[0].isupper():
                author_list.append(token)
    return list(set(author_list))


def extract_entities(sent: str, author_list: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    nlp = _load_hf_pipeline()
    entities: List[Dict[str, Any]] = []

    for ent in nlp(sent):
        label = (ent.get("entity_group") or ent.get("entity") or "").upper()
        score = float(ent.get("score", 1.0))
        text = clean_entity_text(ent.get("word", ""))

        if score < SCORE_THRESH:
            continue
        if should_filter_entity(text, author_list):
            continue

        entities.append(
            {
                "start": int(ent.get("start", -1)),
                "end": int(ent.get("end", -1)),
                "label": label,
                "text": text,
                "score": score,
            }
        )

    return entities


def extract_relations(sent: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chems = [e for e in entities if e["label"].upper() in DRUG_LABELS]
    dises = [e for e in entities if e["label"].upper() in ADR_LABELS]

    if not chems or not dises:
        return []

    rel_type = "causes" if any(re.search(p, sent.lower()) for p in CAUSE_PATTERNS) else "mention"
    pair_max: Dict[tuple, Dict[str, Any]] = {}

    for c in chems:
        for d in dises:
            pair = (c["text"], d["text"])
            avg_conf = (float(c["score"]) + float(d["score"])) / 2.0

            prev = pair_max.get(pair)
            if prev is None or avg_conf > prev["confidence"]:
                pair_max[pair] = {
                    "sentence": sent,
                    "chemical": c["text"],
                    "disease": d["text"],
                    "rel_type": rel_type,
                    "chemical_conf": float(c["score"]),
                    "disease_conf": float(d["score"]),
                    "confidence": float(avg_conf),
                }

    return sorted(pair_max.values(), key=lambda x: -x["confidence"])


def deduplicate_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in relations:
        key = (r["chemical"].lower(), r["disease"].lower(), r["sentence"].strip())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def run_pipeline_with_entities(pdf_bytes: bytes) -> Dict[str, List[Dict[str, Any]]]:
    """
    Input:
      - pdf_bytes: isi file PDF dalam bentuk bytes

    Output:
      - dict:
        entities: list entity level dengan sentence context
        relations: list relations deduplicated
    """
    sentences = extract_pdf_sentences(pdf_bytes)
    if not sentences:
        return {"entities": [], "relations": []}

    author_list = build_author_list(sentences)

    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Dict[str, Any]] = []

    for sent in sentences:
        entities = extract_entities(sent, author_list=author_list)
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

        rels = extract_relations(sent, entities)
        all_relations.extend(rels)

    return {
        "entities": all_entities,
        "relations": deduplicate_relations(all_relations),
    }


def run_pipeline(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Input:
      - pdf_bytes: isi file PDF dalam bentuk bytes

    Output:
      - list of relations:
        chemical, disease, sentence, rel_type, chemical_conf, disease_conf, confidence
    """
    return run_pipeline_with_entities(pdf_bytes)["relations"]