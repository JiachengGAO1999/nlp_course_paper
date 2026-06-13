import random
from pathlib import Path

from src.io_utils import iter_jsonl, read_jsonl, write_json, write_jsonl
from src.musique_sampling import (
    build_candidate,
    hop_distribution,
    sample_by_hop,
    suitability_reason,
    write_preview,
)


def normalize_int_key_map(mapping):
    return {int(k): int(v) for k, v in (mapping or {}).items()}


def read_existing_ids(paths):
    ids = set()
    for path in paths:
        candidate = Path(path)
        if not candidate.exists():
            continue
        for row in read_jsonl(candidate):
            ids.add(row["source_id"])
    return ids


def prepare_formal_pool(config, args):
    seed = int(config["experiment"].get("random_seed", 42))
    layer1 = config.get("layer1") or {}
    allocation = normalize_int_key_map(layer1.get("formal_pool_hop_allocation") or {2: 16, 3: 40, 4: 24})
    min_hops = min(allocation)
    max_hops = max(allocation)
    excluded = set(layer1.get("excluded_source_ids") or [])
    excluded.update(
        read_existing_ids(
            [
                config["paths"].get("smoke_samples"),
                config["paths"].get("pilot_samples"),
            ]
        )
    )

    raw_path = Path(config["paths"]["raw_musique_ans"])
    raw_tmp_path = raw_path.with_suffix(raw_path.suffix + ".tmp")
    readable_raw_path = raw_path if raw_path.exists() else raw_tmp_path

    reasons = {}
    candidates = []
    total = 0
    for item in iter_jsonl(readable_raw_path):
        total += 1
        reason = suitability_reason(item, min_hops, max_hops)
        if reason == "ok" and item.get("id") in excluded:
            reason = "excluded_existing_or_manual"
        reasons[reason] = reasons.get(reason, 0) + 1
        if reason == "ok":
            candidates.append(build_candidate(item))

    sampled = sample_by_hop(candidates, allocation, random.Random(seed + 1009))
    for row in sampled:
        row["split"] = "formal_pool"

    output_path = Path(args.output or config["paths"].get("formal_pool_samples", "data/layer1/splits/formal_pool.jsonl"))
    audit_path = Path(args.audit or "data/layer1/audits/formal_pool_sampling_audit.json")
    preview_path = Path(args.preview or "data/layer1/previews/formal_pool_sampling_preview.md")

    write_jsonl(output_path, sampled)
    write_preview(preview_path, sampled)
    audit = {
        "source": "bdsaglam/musique:musique_ans_v1.0_dev.jsonl",
        "total_rows_seen": total,
        "filter_reasons": dict(sorted(reasons.items())),
        "eligible_count": len(candidates),
        "sampled_count": len(sampled),
        "sampled_hop_distribution": hop_distribution(sampled),
        "excluded_existing_ids_count": len(excluded),
        "output": str(output_path),
        "preview": str(preview_path),
    }
    write_json(audit_path, audit)
    return audit
