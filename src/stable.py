import hashlib
import random


def stable_rng(seed, source_id):
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))
