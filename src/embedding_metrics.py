"""Embedding-based compression quality metrics.

Computes three metric families from the formal run artifacts:
1. Evidence retention: cosine similarity between evidence paragraphs and
   compressed history text.
2. Answer fidelity: cosine similarity between the full_history model answer
   and the compressed-condition model answer.
3. Distractor salience: gold option relevance vs. distractor option relevance
   in the compressed history text.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.io_utils import ensure_dir, read_jsonl, write_json, write_jsonl


MODEL_NAME = "all-MiniLM-L6-v2"


def _embed(texts, model):
    if isinstance(texts, str):
        texts = [texts]
    return model.encode(texts, normalize_embeddings=True)


def _cos_sim(a, b):
    return float(a.flatten() @ b.flatten())


def load_model():
    return SentenceTransformer(MODEL_NAME)


def load_dialogues(path):
    rows = read_jsonl(path)
    return {r["source_id"]: r for r in rows}


def load_variants(path):
    rows = read_jsonl(path)
    index = {}
    for r in rows:
        index.setdefault(r["source_id"], {})[r["condition"]] = r
    return index


def load_generations(path):
    rows = read_jsonl(path)
    index = {}
    for r in rows:
        index.setdefault(r["source_id"], {})[r["condition"]] = r
    return index


def evidence_texts(dialogue_row):
    """Build a single evidence text per sample by joining all required evidence paragraphs."""
    evs = dialogue_row.get("required_evidence") or []
    return "\n\n".join(
        ev.get("paragraph_text", "") or "" for ev in evs
    )


def gold_text(dialogue_row):
    """Gold option text."""
    gold_label = dialogue_row.get("gold")
    options = dialogue_row.get("options") or {}
    return options.get(gold_label, "")


def distractor_texts(dialogue_row):
    """All distractor option texts."""
    gold_label = dialogue_row.get("gold")
    options = dialogue_row.get("options") or {}
    return [(label, text) for label, text in options.items() if label != gold_label]


def answer_text(gen_row):
    """Model's full answer content."""
    return (gen_row.get("response_content") or "").strip()


def reasoning_text(gen_row):
    """Model's reasoning trace."""
    return (gen_row.get("response_reasoning") or "").strip()


def compute_evidence_similarity(evidence, compressed_text, model):
    """Cosine similarity between the full evidence text and compressed history."""
    if not evidence.strip() or not compressed_text.strip():
        return None
    ev_emb = _embed(evidence, model)
    comp_emb = _embed(compressed_text, model)
    return _cos_sim(ev_emb, comp_emb)


def compute_answer_fidelity(fh_answer, comp_answer, model):
    """Cosine similarity between full_history answer and compressed-condition answer."""
    if not fh_answer.strip() or not comp_answer.strip():
        return None
    fh_emb = _embed(fh_answer, model)
    comp_emb = _embed(comp_answer, model)
    return _cos_sim(fh_emb, comp_emb)


def compute_reasoning_similarity(fh_reasoning, comp_reasoning, model):
    """Cosine similarity between full_history reasoning and compressed-condition reasoning."""
    if not fh_reasoning.strip() or not comp_reasoning.strip():
        return None
    fh_emb = _embed(fh_reasoning, model)
    comp_emb = _embed(comp_reasoning, model)
    return _cos_sim(fh_emb, comp_emb)


def compute_distractor_salience(compressed_text, gold_text, distractors, model):
    """Ratio: gold_similarity / mean_distractor_similarity.
    > 1 means gold is more salient than distractors.
    < 1 means distractors are more salient than gold.
    """
    if not compressed_text.strip():
        return None, None
    comp_emb = _embed(compressed_text, model)

    gold_sim = None
    if gold_text.strip():
        gold_emb = _embed(gold_text, model)
        gold_sim = _cos_sim(comp_emb, gold_emb)

    dist_sims = []
    for _label, text in distractors:
        if not text.strip():
            continue
        dist_emb = _embed(text, model)
        dist_sims.append(_cos_sim(comp_emb, dist_emb))

    mean_dist = sum(dist_sims) / len(dist_sims) if dist_sims else None
    ratio = gold_sim / mean_dist if (gold_sim and mean_dist and mean_dist > 0) else None
    return gold_sim, ratio


def compute_all(dialogue_path, variant_path, generation_path, output_path, output_summary_path):
    model = load_model()
    print("Model loaded:", MODEL_NAME)

    dialogues = load_dialogues(dialogue_path)
    variants = load_variants(variant_path)
    generations = load_generations(generation_path)

    rows = []
    for sid, d in dialogues.items():
        evidence = evidence_texts(d)
        gold = gold_text(d)
        distractors = distractor_texts(d)
        fh_gen = generations.get(sid, {}).get("full_history")
        fh_ans = answer_text(fh_gen) if fh_gen else ""
        fh_reasoning = reasoning_text(fh_gen) if fh_gen else ""

        sv = variants.get(sid, {})
        sg = generations.get(sid, {})

        for cond in ["one_shot_summary", "hybrid_summary_recent"]:
            v = sv.get(cond, {})
            g = sg.get(cond, {})
            if not v or not g:
                continue

            comp_text = (v.get("history_text") or "").strip()
            comp_ans = answer_text(g) if g else ""
            comp_reasoning = reasoning_text(g) if g else ""

            ev_sim = compute_evidence_similarity(evidence, comp_text, model)
            ans_fid = compute_answer_fidelity(fh_ans, comp_ans, model)
            reason_sim = compute_reasoning_similarity(fh_reasoning, comp_reasoning, model)
            gold_sim, dist_ratio = compute_distractor_salience(comp_text, gold, distractors, model)

            rows.append({
                "source_id": sid,
                "condition": cond,
                "dialogue_profile": d.get("dialogue_profile"),
                "hop_count": d.get("hop_count"),
                "critical_evidence_in_recent_turn": d.get("critical_evidence_in_recent_turn"),
                "full_history_tokens": v.get("full_history_tokens"),
                "history_tokens": v.get("history_tokens"),
                "is_correct": g.get("is_correct"),
                "parsed_answer": g.get("parsed_answer"),
                "gold": d.get("gold"),
                "evidence_similarity": ev_sim,
                "answer_fidelity": ans_fid,
                "reasoning_similarity": reason_sim,
                "gold_distractor_salience_ratio": dist_ratio,
                "gold_similarity": gold_sim,
            })

    write_jsonl(output_path, rows)

    # Summary: group by condition × profile × crit_rec
    from collections import defaultdict
    groups = defaultdict(list)

    for row in rows:
        key = "{}|{}|crit_rec={}".format(
            row["condition"],
            row["dialogue_profile"],
            row["critical_evidence_in_recent_turn"],
        )
        groups[key].append(row)

    summary = {}
    for key, group in sorted(groups.items()):
        ev_sims = [r["evidence_similarity"] for r in group if r["evidence_similarity"] is not None]
        ans_fids = [r["answer_fidelity"] for r in group if r["answer_fidelity"] is not None]
        reason_sims = [r["reasoning_similarity"] for r in group if r["reasoning_similarity"] is not None]
        ratios = [r["gold_distractor_salience_ratio"] for r in group if r["gold_distractor_salience_ratio"] is not None]
        gold_sims = [r["gold_similarity"] for r in group if r["gold_similarity"] is not None]
        acc = sum(1 for r in group if r["is_correct"]) / len(group) if group else None

        summary[key] = {
            "n": len(group),
            "accuracy": acc,
            "evidence_similarity_mean": sum(ev_sims) / len(ev_sims) if ev_sims else None,
            "answer_fidelity_mean": sum(ans_fids) / len(ans_fids) if ans_fids else None,
            "reasoning_similarity_mean": sum(reason_sims) / len(reason_sims) if reason_sims else None,
            "gold_distractor_ratio_mean": sum(ratios) / len(ratios) if ratios else None,
            "gold_similarity_mean": sum(gold_sims) / len(gold_sims) if gold_sims else None,
        }

    write_json(output_summary_path, summary)
    print("Wrote {} metric rows to {}".format(len(rows), output_path))
    print("Wrote summary to {}".format(output_summary_path))
    return {"rows": len(rows), "output": str(output_path), "summary": str(output_summary_path)}
