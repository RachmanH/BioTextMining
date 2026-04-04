from collections import defaultdict
from typing import Any, Dict, List, Tuple


def _norm_text(value: Any) -> str:
    # Normalisasi dasar agar key agregasi konsisten.
    return str(value or "").strip().lower()


def aggregate_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input:
      relations = list of dict dari pipeline, minimal berisi:
      - chemical
      - disease
      - confidence
      - sentence

    Output:
      list agregasi:
      - chemical
      - disease
      - count
      - avg_conf
      - sentences (maks 3 evidence unik)
    """
    agg: Dict[Tuple[str, str], Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "conf": [], "sentences": []}
    )

    for r in relations:
        chem = _norm_text(r.get("chemical"))
        dis = _norm_text(r.get("disease"))

        # Skip relasi tidak valid supaya DB bersih.
        if not chem or not dis:
            continue

        key = (chem, dis)

        conf = r.get("confidence", 0.0)
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 0.0

        sent = str(r.get("sentence") or "").strip()

        agg[key]["count"] += 1
        agg[key]["conf"].append(conf)

        # Simpan sentence unik agar evidence tidak duplikat.
        if sent and sent not in agg[key]["sentences"]:
            agg[key]["sentences"].append(sent)

    results: List[Dict[str, Any]] = []
    for (chem, dis), v in agg.items():
        confs = v["conf"] or [0.0]
        results.append(
            {
                "chemical": chem,
                "disease": dis,
                "count": int(v["count"]),
                "avg_conf": float(sum(confs) / len(confs)),
                "sentences": v["sentences"][:3],
            }
        )

    # Optional: urutkan agar hasil stabil (terbesar dulu).
    results.sort(key=lambda x: (x["count"], x["avg_conf"]), reverse=True)
    return results