import argparse
import json
import os
import random
import statistics
import sys
from pathlib import Path
from urllib.request import Request, urlopen

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


MUSIQUE_ANS_DEV_URL = (
    "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/"
    "musique_ans_v1.0_dev.jsonl"
)


def load_yaml(path):
    if yaml is None:
        raise RuntimeError("PyYAML is required to read configs/experiment.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def download_if_missing(url, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return False
    ensure_dir(output_path.parent)
    headers = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    with urlopen(req, timeout=120) as response, open(output_path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return True


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}") from exc


def answer_is_simple(answer):
    text = str(answer or "").strip()
    if not text:
        return False
    lowered = text.lower()
    if lowered in {"yes", "no", "true", "false"}:
        return False
    if len(text.split()) > 6:
        return False
    if len(text) > 80:
        return False
    return True


def get_supporting_paragraphs(item):
    paragraphs = item.get("paragraphs") or []
    return [p for p in paragraphs if p.get("is_supporting")]


def get_required_evidence(item):
    paragraphs = item.get("paragraphs") or []
    evidence = []
    for step_idx, step in enumerate(item.get("question_decomposition") or [], start=1):
        paragraph_idx = step.get("paragraph_support_idx")
        if paragraph_idx is None:
            return None
        if not isinstance(paragraph_idx, int) or paragraph_idx < 0 or paragraph_idx >= len(paragraphs):
            return None
        paragraph = paragraphs[paragraph_idx]
        if not paragraph.get("is_supporting"):
            return None
        evidence.append(
            {
                "step": step_idx,
                "subquestion": step.get("question"),
                "subanswer": step.get("answer"),
                "paragraph_idx": paragraph_idx,
                "title": paragraph.get("title"),
                "paragraph_text": paragraph.get("paragraph_text"),
            }
        )
    return evidence


def suitability_reason(item, min_hops, max_hops):
    if not item.get("answerable"):
        return "not_answerable"

    decomposition = item.get("question_decomposition") or []
    hop_count = len(decomposition)
    if hop_count < min_hops or hop_count > max_hops:
        return "hop_count_out_of_range"

    if not answer_is_simple(item.get("answer")):
        return "answer_not_simple"

    supporting = get_supporting_paragraphs(item)
    if len(supporting) != hop_count:
        return "supporting_count_mismatch"

    evidence = get_required_evidence(item)
    if evidence is None or len(evidence) != hop_count:
        return "missing_or_invalid_step_support"

    return "ok"


def build_candidate(item):
    evidence = get_required_evidence(item)
    return {
        "source": "MuSiQue-Ans",
        "source_id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "answer_aliases": item.get("answer_aliases") or [],
        "hop_count": len(item.get("question_decomposition") or []),
        "required_evidence": evidence,
        "paragraph_count": len(item.get("paragraphs") or []),
        "supporting_paragraph_count": len(get_supporting_paragraphs(item)),
        "raw_question_decomposition": item.get("question_decomposition") or [],
        "mc_conversion_status": "pending",
        "dialogue_generation_status": "pending",
    }


def normalize_int_key_map(mapping):
    return {int(k): int(v) for k, v in (mapping or {}).items()}


def get_layer1_allocations(config):
    layer1 = config.get("layer1") or {}
    hop_allocation = normalize_int_key_map(layer1.get("hop_allocation"))
    split_hop_allocation = {
        name: normalize_int_key_map(allocation)
        for name, allocation in (layer1.get("split_hop_allocation") or {}).items()
    }
    return hop_allocation, split_hop_allocation


def get_excluded_source_ids(config):
    layer1 = config.get("layer1") or {}
    return set(layer1.get("excluded_source_ids") or [])


def sample_by_hop(candidates, hop_allocation, rng):
    by_hop = {}
    for candidate in candidates:
        by_hop.setdefault(candidate["hop_count"], []).append(candidate)

    selected = []
    for hop_count, count in sorted(hop_allocation.items()):
        pool = list(by_hop.get(hop_count, []))
        if len(pool) < count:
            raise ValueError(
                f"Need {count} eligible {hop_count}-hop samples, but only found {len(pool)}"
            )
        rng.shuffle(pool)
        selected.extend(pool[:count])

    rng.shuffle(selected)
    return selected


def split_by_hop(sampled, split_hop_allocation, rng):
    by_hop = {}
    for candidate in sampled:
        by_hop.setdefault(candidate["hop_count"], []).append(candidate)
    for pool in by_hop.values():
        rng.shuffle(pool)

    splits = {}
    used_ids = set()
    for split_name in ["smoke", "pilot", "formal", "spares"]:
        allocation = split_hop_allocation.get(split_name, {})
        rows = []
        for hop_count, count in sorted(allocation.items()):
            available = [row for row in by_hop.get(hop_count, []) if row["source_id"] not in used_ids]
            if len(available) < count:
                raise ValueError(
                    f"Need {count} {hop_count}-hop samples for {split_name}, "
                    f"but only {len(available)} remain"
                )
            chosen = available[:count]
            for row in chosen:
                row["split"] = split_name
                used_ids.add(row["source_id"])
            rows.extend(chosen)
        rng.shuffle(rows)
        splits[split_name] = rows

    expected_ids = {row["source_id"] for row in sampled}
    if used_ids != expected_ids:
        missing = sorted(expected_ids - used_ids)
        raise ValueError(f"Split allocations did not consume all sampled rows: {missing[:5]}")
    return splits


def write_jsonl(path, rows):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_preview(path, candidates):
    ensure_dir(path.parent)
    lines = [
        "# MuSiQue Layer 1 Candidate Preview",
        "",
        "Use this file for quick manual sanity checks before MC conversion.",
        "",
    ]
    for idx, item in enumerate(candidates[:12], start=1):
        lines.append(f"## {idx}. {item['source_id']}")
        lines.append("")
        lines.append(f"Question: {item['question']}")
        lines.append("")
        lines.append(f"Answer: {item['answer']}")
        lines.append("")
        lines.append(f"Hops: {item['hop_count']}")
        lines.append("")
        for ev in item["required_evidence"]:
            paragraph = ev["paragraph_text"].replace("\n", " ")
            if len(paragraph) > 420:
                paragraph = paragraph[:417] + "..."
            lines.append(
                f"- Step {ev['step']}: {ev['subquestion']} -> {ev['subanswer']} "
                f"[{ev['title']}]"
            )
            lines.append(f"  Evidence: {paragraph}")
        lines.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")


def hop_distribution(rows):
    counts = {}
    for row in rows:
        hop_count = str(row["hop_count"])
        counts[hop_count] = counts.get(hop_count, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def summarize(all_count, reasons, candidates, sampled, splits):
    hop_counts = [c["hop_count"] for c in candidates]
    evidence_counts = [len(c["required_evidence"]) for c in candidates]
    answer_words = [len(str(c["answer"]).split()) for c in candidates]
    return {
        "source": "bdsaglam/musique:musique_ans_v1.0_dev.jsonl",
        "total_rows_seen": all_count,
        "filter_reasons": dict(sorted(reasons.items())),
        "eligible_count": len(candidates),
        "sampled_count": len(sampled),
        "hop_count_distribution": {
            str(h): sum(1 for x in hop_counts if x == h) for h in sorted(set(hop_counts))
        },
        "sampled_hop_distribution": hop_distribution(sampled),
        "split_hop_distribution": {
            name: hop_distribution(rows) for name, rows in splits.items()
        },
        "eligible_required_evidence": {
            "min": min(evidence_counts) if evidence_counts else None,
            "mean": statistics.mean(evidence_counts) if evidence_counts else None,
            "max": max(evidence_counts) if evidence_counts else None,
        },
        "eligible_answer_words": {
            "min": min(answer_words) if answer_words else None,
            "mean": statistics.mean(answer_words) if answer_words else None,
            "max": max(answer_words) if answer_words else None,
        },
        "sample_ids": [c["source_id"] for c in sampled],
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--split", default="dev", choices=["dev"])
    parser.add_argument("--min-hops", type=int, default=None)
    parser.add_argument("--max-hops", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    seed = int(config["experiment"].get("random_seed", 42))
    hop_allocation, split_hop_allocation = get_layer1_allocations(config)
    excluded_source_ids = get_excluded_source_ids(config)
    if not hop_allocation:
        num_samples = args.num_samples or 60
        hop_allocation = {2: num_samples}
    min_hops = args.min_hops if args.min_hops is not None else min(hop_allocation)
    max_hops = args.max_hops if args.max_hops is not None else max(hop_allocation)
    data_dir = Path(config["paths"].get("data_dir", "data"))
    raw_path = Path(config["paths"].get("raw_musique_ans", data_dir / "raw" / "musique_ans_v1.0_dev.jsonl"))
    raw_tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    candidates_path = Path(config["paths"].get("layer1_candidates", data_dir / "layer1" / "splits" / "candidates.jsonl"))
    audit_path = Path(config["paths"].get("layer1_audit", data_dir / "layer1" / "audits" / "musique_sampling_audit.json"))
    preview_path = Path(config["paths"].get("layer1_preview", data_dir / "layer1" / "previews" / "musique_sampling_preview.md"))
    spares_path = Path(config["paths"].get("layer1_spares", data_dir / "layer1" / "splits" / "spares.jsonl"))

    if raw_path.exists() or raw_tmp_path.exists():
        downloaded = False
    else:
        downloaded = download_if_missing(MUSIQUE_ANS_DEV_URL, raw_path)
    readable_raw_path = raw_path if raw_path.exists() else raw_tmp_path

    reasons = {}
    candidates = []
    all_count = 0
    for item in iter_jsonl(readable_raw_path):
        all_count += 1
        reason = suitability_reason(item, min_hops, max_hops)
        if reason == "ok" and item.get("id") in excluded_source_ids:
            reason = "manually_excluded"
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "ok":
            candidates.append(build_candidate(item))

    rng = random.Random(seed)
    sampled = sample_by_hop(candidates, hop_allocation, rng)
    splits = split_by_hop(sampled, split_hop_allocation, rng)

    write_jsonl(candidates_path, sampled)
    write_jsonl(Path(config["paths"]["smoke_samples"]), splits["smoke"])
    write_jsonl(Path(config["paths"]["pilot_samples"]), splits["pilot"])
    write_jsonl(Path(config["paths"]["formal_samples"]), splits["formal"])
    write_jsonl(spares_path, splits["spares"])
    audit = summarize(all_count, reasons, candidates, sampled, splits)
    audit["downloaded_raw_file"] = downloaded
    audit["raw_path"] = str(readable_raw_path)
    audit["candidates_path"] = str(candidates_path)
    audit["smoke_path"] = config["paths"]["smoke_samples"]
    audit["pilot_path"] = config["paths"]["pilot_samples"]
    audit["formal_path"] = config["paths"]["formal_samples"]
    audit["spares_path"] = str(spares_path)
    audit["preview_path"] = str(preview_path)
    write_json(audit_path, audit)
    write_preview(preview_path, sampled)

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
