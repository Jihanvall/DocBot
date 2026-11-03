import hashlib
from pathlib import Path

import numpy as np

CACHE_DIR = Path(".cache/embeddings")


def cache_key(texts, model, task_type):
    h = hashlib.sha256()
    h.update(f"{model}|{task_type}".encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:20]


def cached_embed(embed, texts, model, task_type, cache_dir=None):
    folder = Path(cache_dir) if cache_dir else CACHE_DIR
    path = folder / f"{cache_key(texts, model, task_type)}.npy"
    if path.exists():
        return np.load(path)
    vectors = embed(texts, task_type)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors
