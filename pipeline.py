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
ABSTRACT_WEIGHT = float(os.getenv("ABSTRACT_WEIGHT", "0.85"))
DEFAULT_SECTION_WEIGHT = float(os.getenv("DEFAULT_SECTION_WEIGHT", "1.0"))

CAUSE_PATTERNS = [
    r"cause(s|d)?",
    r"induce(s|d)?",
    r"lead(s|ing)? to",
    r"triggers?",
    r"is associated with",
    r"results? in",
    r"contributes? to",
]

STRONG_CAUSE_PATTERNS = [
    r"\bcaus(?:e|es|ed|ing)\b",
    r"\binduc(?:e|es|ed|ing)\b",
    r"\btrigger(?:s|ed|ing)?\b",
    r"\blead(?:s|ing)? to\b",
    r"\bresult(?:s|ed|ing)? in\b",
    r"\bbring(?:s|ing)? about\b",
    r"\bresponsible for\b",
    r"\bcontribut(?:e|es|ed|ing) to\b",
]

WEAK_ASSOCIATION_PATTERNS = [
    r"\bassociat(?:e|es|ed|ing) with\b",
    r"\blink(?:s|ed|ing)? to\b",
    r"\bcorrelat(?:e|es|ed|ing) with\b",
]

NEGATION_PATTERNS = [
    r"\bno evidence of\b",
    r"\bno sign of\b",
    r"\bnot\s+(?:cause|causes|caused|causing|induce|induced|inducing|trigger|triggered|triggering|lead to|leads to|leading to|result in|results in|resulting in)\b",
    r"\bwithout\b",
    r"\babsence of\b",
    r"\bnot associated with\b",
    r"\bnon[- ]associated\b",
]

SPECULATION_PATTERNS = [
    r"\bmay\s+(?:cause|causes|caused|causing|lead to|leads to|leading to|induce|induces|induced|inducing|trigger|triggers|triggered|triggering|result in|results in|resulting in|contribute to|contributes to|contributing to)\b",
    r"\bmight\s+(?:cause|lead to|induce|trigger|result in|contribute to)\b",
    r"\bcould\s+(?:cause|lead to|induce|trigger|result in|contribute to)\b",
    r"\bcan\s+(?:cause|lead to|induce|trigger|result in|contribute to)\b",
    r"\bpossible(?:ly)?\b",
    r"\bpotentially\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bsuggest(?:s|ed|ing)?\b",
    r"\bappears? to\b",
    r"\bseems? to\b",
]

RELATION_MAX_TOKEN_GAP = int(os.getenv("RELATION_MAX_TOKEN_GAP", "18"))
RELATION_CONF_THRESH = float(os.getenv("RELATION_CONF_THRESH", "0.50"))
RELATION_MIN_SUPPORT = int(os.getenv("RELATION_MIN_SUPPORT", "1"))

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

SECTION_ALIASES = {
    "abstract": "abstract",
    "methods": "methods",
    "material and methods": "methods",
    "materials and methods": "methods",
    "results": "results",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "references": "references",
    "reference": "references",
}

SECTION_HEADING_RE = re.compile(
    r"^\s*(abstract|methods?|materials? and methods?|results?|discussion|conclusions?|references?|reference)\s*[:\.]?\s*$",
    re.IGNORECASE,
)


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


def _normalize_section_name(section_name: Optional[str]) -> str:
    """Map a raw heading to a canonical section name."""
    normalized = str(section_name or "").strip().lower()
    return SECTION_ALIASES.get(normalized, normalized)


def _detect_section_heading(line: str) -> Optional[str]:
    """Detect simple section headings from PDF lines."""
    match = SECTION_HEADING_RE.match(str(line or "").strip())
    if not match:
        return None

    heading = match.group(1).strip().lower()
    return _normalize_section_name(heading)


def _section_weight(section_name: str) -> float:
    """Assign trust weight to a section before NER and relation extraction."""
    section_name = _normalize_section_name(section_name)
    if section_name == "abstract":
        return ABSTRACT_WEIGHT
    if section_name == "results":
        return 1.0
    return DEFAULT_SECTION_WEIGHT


def _should_skip_section(section_name: str) -> bool:
    """Skip sections that are too noisy for relation extraction."""
    section_name = _normalize_section_name(section_name)
    return section_name in {"methods", "references"}


def _has_any_pattern(text: str, patterns: List[str]) -> bool:
    lowered = str(text or "").lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)


def _token_distance(sentence: str, left: Dict[str, Any], right: Dict[str, Any]) -> int:
    start_left = int(left.get("start", -1))
    end_left = int(left.get("end", -1))
    start_right = int(right.get("start", -1))
    end_right = int(right.get("end", -1))

    if start_left < 0 or end_left < 0 or start_right < 0 or end_right < 0:
        return 10**6

    if start_left > start_right:
        start_left, end_left, start_right, end_right = start_right, end_right, start_left, end_left

    between = str(sentence or "")[end_left:start_right]
    if not between.strip():
        return 0

    return len([token for token in re.split(r"\s+", between.strip()) if token])


def _is_strict_causal_sentence(sentence: str) -> bool:
    lowered = str(sentence or "").lower()
    if not lowered:
        return False
    if _has_any_pattern(lowered, NEGATION_PATTERNS):
        return False
    if _has_any_pattern(lowered, SPECULATION_PATTERNS):
        return False
    if _has_any_pattern(lowered, STRONG_CAUSE_PATTERNS):
        return True
    if _has_any_pattern(lowered, WEAK_ASSOCIATION_PATTERNS):
        return True
    return False


def _boost_relation_confidence(base_confidence: float, sentence: str) -> float:
    confidence = float(base_confidence)
    if _has_any_pattern(sentence, STRONG_CAUSE_PATTERNS):
        confidence += 0.08
    elif _has_any_pattern(sentence, WEAK_ASSOCIATION_PATTERNS):
        confidence += 0.03
    if _has_any_pattern(sentence, [r"\bresults? in\b", r"\bleads? to\b", r"\bcauses?\b"]):
        confidence += 0.04
    return min(confidence, 0.999)


def _aggregate_relation_evidence(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, Dict[str, Any]] = {}

    for rel in relations:
        chemical = str(rel.get("chemical", "")).strip()
        disease = str(rel.get("disease", "")).strip()
        sentence = str(rel.get("sentence", "")).strip()
        if not chemical or not disease or not sentence:
            continue

        key = (chemical.lower(), disease.lower())
        bucket = grouped.setdefault(
            key,
            {
                "chemical": chemical,
                "disease": disease,
                "count": 0,
                "conf_sum": 0.0,
                "best_confidence": -1.0,
                "sentence": sentence,
                "sentences": [],
                "rel_type": "causes",
                "chemical_conf": float(rel.get("chemical_conf", 0.0)),
                "disease_conf": float(rel.get("disease_conf", 0.0)),
            },
        )

        confidence = float(rel.get("confidence", 0.0))
        bucket["count"] += 1
        bucket["conf_sum"] += confidence

        if sentence not in bucket["sentences"]:
            bucket["sentences"].append(sentence)

        if confidence >= bucket["best_confidence"]:
            bucket["best_confidence"] = confidence
            bucket["sentence"] = sentence
            bucket["chemical_conf"] = float(rel.get("chemical_conf", bucket["chemical_conf"]))
            bucket["disease_conf"] = float(rel.get("disease_conf", bucket["disease_conf"]))

    aggregated: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        if bucket["count"] < RELATION_MIN_SUPPORT:
            continue

        avg_conf = bucket["conf_sum"] / max(bucket["count"], 1)
        if avg_conf < RELATION_CONF_THRESH:
            continue

        aggregated.append(
            {
                "sentence": bucket["sentence"],
                "chemical": bucket["chemical"],
                "disease": bucket["disease"],
                "rel_type": bucket["rel_type"],
                "chemical_conf": float(bucket["chemical_conf"]),
                "disease_conf": float(bucket["disease_conf"]),
                "confidence": float(avg_conf),
                "support_count": int(bucket["count"]),
                "evidence_sentences": bucket["sentences"][:3],
            }
        )

    return sorted(aggregated, key=lambda item: (-float(item.get("confidence", 0.0)), -int(item.get("support_count", 0))))


def _extract_sentence_units_from_pdf_text(raw_text: str) -> List[Dict[str, Any]]:
    """
    Extract sentence units with section context.

    References are dropped entirely, Methods are skipped, Abstract is kept but downweighted,
    and Results are kept as the primary target section.
    """
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(raw_text or "").splitlines()]

    units: List[Dict[str, Any]] = []
    current_section = "unknown"
    buffer: List[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return

        block = " ".join(buffer).strip()
        buffer = []
        if not block or _should_skip_section(current_section):
            return

        weight = _section_weight(current_section)
        sentences = nltk.sent_tokenize(block)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) <= 10:
                continue
            units.append(
                {
                    "sentence": sentence,
                    "section": _normalize_section_name(current_section),
                    "weight": weight,
                }
            )

    for line in lines:
        if not line:
            continue

        heading = _detect_section_heading(line)
        if heading:
            flush_buffer()
            current_section = heading
            continue

        # Heuristic: preserve content until a new heading appears.
        buffer.append(line)

    flush_buffer()

    # If no headings are detected, fall back to normal sentence splitting with default weight.
    if units:
        return units

    fallback_sentences = nltk.sent_tokenize(re.sub(r"\s+", " ", str(raw_text or "")).strip())
    return [
        {
            "sentence": sentence.strip(),
            "section": "unknown",
            "weight": DEFAULT_SECTION_WEIGHT,
        }
        for sentence in fallback_sentences
        if len(sentence.strip()) > 10
    ]


def extract_pdf_sentences(pdf_bytes: bytes) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []
    for page in doc:
        all_text.append(page.get_text("text"))
    raw_text = "\n".join(all_text)

    raw_text = re.sub(r"\s+", " ", raw_text).strip()

    if not raw_text:
        return []

    sentence_units = _extract_sentence_units_from_pdf_text(raw_text)
    return [str(unit.get("sentence", "")).strip() for unit in sentence_units if str(unit.get("sentence", "")).strip()]


def build_author_list(sentences: List[str], take_n: int = 5) -> List[str]:
    author_list: List[str] = []
    for s in sentences[:take_n]:
        parts = re.split(r",|and", s)
        for p in parts:
            token = p.strip()
            if 2 < len(token) < 30 and token[0].isupper():
                author_list.append(token)
    return list(set(author_list))


def extract_entities(
    sent: str,
    author_list: Optional[List[str]] = None,
    section_weight: float = DEFAULT_SECTION_WEIGHT,
) -> List[Dict[str, Any]]:
    nlp = _load_hf_pipeline()
    entities: List[Dict[str, Any]] = []

    for ent in nlp(sent):
        label = (ent.get("entity_group") or ent.get("entity") or "").upper()
        score = float(ent.get("score", 1.0)) * float(section_weight)
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


def inspect_relation_sentence(sent: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Inspect one sentence and explain how many relation pairs were dropped by each rule."""
    chems = [e for e in entities if e["label"].upper() in DRUG_LABELS]
    dises = [e for e in entities if e["label"].upper() in ADR_LABELS]

    summary: Dict[str, Any] = {
        "sentence": sent,
        "chem_count": len(chems),
        "disease_count": len(dises),
        "strict_causal": _is_strict_causal_sentence(sent),
        "candidate_pairs": len(chems) * len(dises),
        "accepted_pairs": 0,
        "drop_no_chemical": 0,
        "drop_no_disease": 0,
        "drop_non_causal": 0,
        "drop_bad_span": 0,
        "drop_direction": 0,
        "drop_token_gap": 0,
        "drop_low_conf": 0,
        "max_token_gap": RELATION_MAX_TOKEN_GAP,
        "confidence_threshold": RELATION_CONF_THRESH,
    }

    if not chems:
        summary["drop_no_chemical"] = 1
        return summary

    if not dises:
        summary["drop_no_disease"] = 1
        return summary

    if not summary["strict_causal"]:
        summary["drop_non_causal"] = summary["candidate_pairs"]
        return summary

    for c in chems:
        for d in dises:
            c_start = int(c.get("start", -1))
            d_start = int(d.get("start", -1))
            if c_start < 0 or d_start < 0:
                summary["drop_bad_span"] += 1
                continue

            if c_start >= d_start:
                summary["drop_direction"] += 1
                continue

            token_gap = _token_distance(sent, c, d)
            if token_gap > RELATION_MAX_TOKEN_GAP:
                summary["drop_token_gap"] += 1
                continue

            avg_conf = (float(c["score"]) + float(d["score"])) / 2.0
            avg_conf = _boost_relation_confidence(avg_conf, sent)
            if avg_conf < RELATION_CONF_THRESH:
                summary["drop_low_conf"] += 1
                continue

            summary["accepted_pairs"] += 1

    return summary


def extract_relations(sent: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chems = [e for e in entities if e["label"].upper() in DRUG_LABELS]
    dises = [e for e in entities if e["label"].upper() in ADR_LABELS]

    if not chems or not dises:
        return []

    if not _is_strict_causal_sentence(sent):
        return []

    pair_max: Dict[tuple, Dict[str, Any]] = {}
    ordered_chems = sorted(chems, key=lambda item: (int(item.get("start", -1)), int(item.get("end", -1))))
    ordered_dises = sorted(dises, key=lambda item: (int(item.get("start", -1)), int(item.get("end", -1))))

    for c in ordered_chems:
        for d in ordered_dises:
            if int(c.get("start", -1)) < 0 or int(d.get("start", -1)) < 0:
                continue
            if int(c.get("start", -1)) >= int(d.get("start", -1)):
                continue

            token_gap = _token_distance(sent, c, d)
            if token_gap > RELATION_MAX_TOKEN_GAP:
                continue

            pair = (c["text"], d["text"])
            avg_conf = (float(c["score"]) + float(d["score"])) / 2.0
            avg_conf = _boost_relation_confidence(avg_conf, sent)

            if avg_conf < RELATION_CONF_THRESH:
                continue

            prev = pair_max.get(pair)
            if prev is None or avg_conf > prev["confidence"]:
                pair_max[pair] = {
                    "sentence": sent,
                    "chemical": c["text"],
                    "disease": d["text"],
                    "rel_type": "causes",
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
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []
    for page in doc:
        all_text.append(page.get_text("text"))
    raw_text = "\n".join(all_text).strip()

    if not raw_text:
        return {"entities": [], "relations": []}

    sentence_units = _extract_sentence_units_from_pdf_text(raw_text)
    if not sentence_units:
        return {"entities": [], "relations": []}

    author_list = build_author_list([unit["sentence"] for unit in sentence_units])

    all_entities: List[Dict[str, Any]] = []
    all_relations: List[Dict[str, Any]] = []

    for unit in sentence_units:
        sent = str(unit.get("sentence", "")).strip()
        section_name = _normalize_section_name(unit.get("section", "unknown"))
        section_weight = float(unit.get("weight", DEFAULT_SECTION_WEIGHT))

        entities = extract_entities(sent, author_list=author_list, section_weight=section_weight)
        for e in entities:
            all_entities.append(
                {
                    "sentence": sent,
                    "section": section_name,
                    "start": int(e.get("start", -1)),
                    "end": int(e.get("end", -1)),
                    "label": str(e.get("label", "")),
                    "text": str(e.get("text", "")),
                    "score": float(e.get("score", 0.0)),
                }
            )

        rels = extract_relations(sent, entities)
        all_relations.extend(rels)

    deduped_relations = deduplicate_relations(all_relations)

    return {
        "entities": all_entities,
        "relations": _aggregate_relation_evidence(deduped_relations),
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