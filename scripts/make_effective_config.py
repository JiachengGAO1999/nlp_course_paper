import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io_utils import ensure_dir, load_yaml

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def normalize_int_key_map(mapping):
    return {int(k): int(v) for k, v in (mapping or {}).items()}


def normalize_str_key_map(mapping):
    return {str(k): int(v) for k, v in (mapping or {}).items()}


def scale_allocation(allocation, target_total):
    current_total = sum(allocation.values())
    if current_total <= 0:
        raise ValueError("Cannot scale an empty allocation")
    if current_total == target_total:
        return dict(allocation)

    scaled = {}
    remainders = []
    for key, value in allocation.items():
        exact = value * target_total / current_total
        base = int(exact)
        scaled[key] = base
        remainders.append((exact - base, key))

    remaining = target_total - sum(scaled.values())
    for _, key in sorted(remainders, reverse=True)[:remaining]:
        scaled[key] += 1
    return scaled


def parse_json_map(value, normalize):
    if not value:
        return None
    return normalize(json.loads(value))


def stringify_keys(mapping):
    return {str(k): int(v) for k, v in mapping.items()}


def update_split_allocation(config, split_key, split_name, allocation):
    split_alloc = config.setdefault("layer1", {}).setdefault(split_key, {})
    split_alloc[split_name] = stringify_keys(allocation)


def make_effective_config(config, args):
    effective = copy.deepcopy(config)
    layer1 = effective.setdefault("layer1", {})
    data = effective.setdefault("data", {})
    overrides = {}

    if args.model:
        effective.setdefault("model", {})["model"] = args.model
        effective.setdefault("summarizer", {})["model"] = args.model
        overrides["model"] = args.model
    if args.base_url:
        effective.setdefault("model", {})["base_url"] = args.base_url
        effective.setdefault("summarizer", {})["base_url"] = args.base_url
        overrides["base_url"] = args.base_url
    if args.server_version:
        effective.setdefault("model", {})["server_version"] = args.server_version
        overrides["server_version"] = args.server_version
    if args.model_extra_body_json is not None:
        extra_body = json.loads(args.model_extra_body_json) if args.model_extra_body_json else {}
        effective.setdefault("model", {})["extra_body"] = extra_body
        overrides["model_extra_body"] = extra_body

    pool_size = int(args.formal_pool_size) if args.formal_pool_size is not None else None
    target_n = int(args.formal_target_n) if args.formal_target_n is not None else None

    pool_hop = parse_json_map(args.formal_pool_hop_allocation_json, normalize_int_key_map)
    if pool_hop is None:
        pool_hop = normalize_int_key_map(layer1.get("formal_pool_hop_allocation") or {})
        if pool_size is not None:
            pool_hop = scale_allocation(pool_hop, pool_size)
    if pool_size is None and pool_hop:
        pool_size = sum(pool_hop.values())
    if pool_size is not None:
        layer1["formal_pool_size"] = pool_size
        overrides["formal_pool_size"] = pool_size
    if pool_hop:
        layer1["formal_pool_hop_allocation"] = stringify_keys(pool_hop)
        update_split_allocation(effective, "split_hop_allocation", "formal_pool", pool_hop)
        overrides["formal_pool_hop_allocation"] = stringify_keys(pool_hop)

    formal_hop = parse_json_map(args.formal_hop_allocation_json, normalize_int_key_map)
    if formal_hop is None:
        formal_hop = normalize_int_key_map(layer1.get("formal_hop_allocation") or {})
        if target_n is not None:
            formal_hop = scale_allocation(formal_hop, target_n)
    if target_n is None and formal_hop:
        target_n = sum(formal_hop.values())
    if target_n is not None:
        data["formal_num_samples"] = target_n
        overrides["formal_num_samples"] = target_n
    if formal_hop:
        layer1["formal_hop_allocation"] = stringify_keys(formal_hop)
        update_split_allocation(effective, "split_hop_allocation", "formal", formal_hop)
        overrides["formal_hop_allocation"] = stringify_keys(formal_hop)

    formal_profile = parse_json_map(args.formal_profile_allocation_json, normalize_str_key_map)
    if formal_profile is None:
        formal_profile = normalize_str_key_map(layer1.get("formal_profile_allocation") or {})
        if target_n is not None:
            formal_profile = scale_allocation(formal_profile, target_n)
    if formal_profile:
        layer1["formal_profile_allocation"] = stringify_keys(formal_profile)
        update_split_allocation(effective, "split_profile_allocation", "formal", formal_profile)
        overrides["formal_profile_allocation"] = stringify_keys(formal_profile)

    if pool_size is not None:
        base_pool_profile = normalize_str_key_map(
            (layer1.get("split_profile_allocation") or {}).get("formal_pool")
            or layer1.get("formal_profile_allocation")
            or {}
        )
        if base_pool_profile:
            pool_profile = scale_allocation(base_pool_profile, pool_size)
            update_split_allocation(effective, "split_profile_allocation", "formal_pool", pool_profile)
            overrides["formal_pool_profile_allocation"] = stringify_keys(pool_profile)

    effective["run_overrides"] = {
        "source": "scripts/make_effective_config.py",
        "base_config": args.config,
        "applied": overrides,
    }
    return effective


def write_yaml(path, obj):
    if yaml is None:
        raise RuntimeError("PyYAML is required to write YAML configuration files")
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--formal-pool-size", type=int, default=None)
    parser.add_argument("--formal-target-n", type=int, default=None)
    parser.add_argument("--formal-pool-hop-allocation-json", default=None)
    parser.add_argument("--formal-hop-allocation-json", default=None)
    parser.add_argument("--formal-profile-allocation-json", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--server-version", default=None)
    parser.add_argument("--model-extra-body-json", default=None)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    effective = make_effective_config(config, args)
    write_yaml(args.output, effective)
    print(json.dumps(effective.get("run_overrides", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
