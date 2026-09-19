import csv
import hashlib
import json
import os
import random
from pathlib import Path

def audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    from math import gcd
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        divisor = gcd(sr, 16000)
        x = resample_poly(x, 16000 // divisor, sr // divisor).astype(np.float32)
    if len(x) < 400 or not np.isfinite(x).all():
        raise ValueError(f"Invalid/too short audio: {path}")
    return x


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_rows(split, data):
    with (Path(data) / 'statistic' / f"{split.title()}.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        assert int(r["n_valid"]) == 5 and int(r["n_broken"]) == 0
        vs = [int(float(r[f"v{i}"])) for i in range(1, 6)]
        assert all(-3 <= v <= 3 for v in vs)
        assert abs(sum(vs) / 5 - float(r["mean_strength"])) < 1e-7
        assert [sum(v > 0 for v in vs), sum(v < 0 for v in vs), vs.count(0)] == [
            int(r[k]) for k in ("n_a", "n_b", "n_tie")]
    assert len({r["item"] for r in rows}) == len(rows)
    return rows


def speaker_split(rows):
    # Fixed speaker-disjoint split for checkpoint selection.
    speakers = sorted({r["speaker"] for r in rows})
    random.Random(42).shuffle(speakers)
    held, count = set(), 0
    for speaker in speakers:
        if count >= round(0.2 * len(rows)):
            break
        held.add(speaker)
        count += sum(r["speaker"] == speaker for r in rows)
    train = [i for i, r in enumerate(rows) if r["speaker"] not in held]
    dev = [i for i, r in enumerate(rows) if r["speaker"] in held]
    assert not ({rows[i]["speaker"] for i in train} & {rows[i]["speaker"] for i in dev})
    return train, dev
