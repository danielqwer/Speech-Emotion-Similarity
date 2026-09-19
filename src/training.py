import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from utils import atomic_json
from model import SESJudge, ordinal_loss

EPOCHS = 100
PATIENCE = 15
BATCH = 64
WEIGHT_DECAY = 1e-3


def save_torch(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def tensors(rows, pos, vectors):
    idx = [[pos[r[k]] for k in ("path_ref", "path_a", "path_b")] for r in rows]
    hist = [[sum(int(float(r[f"v{i}"])) == k for i in range(1, 6))/5 for k in range(-3, 4)] for r in rows]
    indices = torch.tensor(idx, device="cuda")
    return vectors[indices], torch.tensor(hist, device="cuda", dtype=torch.float32)


@torch.inference_mode()
def evaluate(model, features):
    model.eval()
    cols = {k: [] for k in ("score", "margin", "sim_a", "sim_b")}
    for i in range(0, len(features), 256):
        out = model(features[i:i+256])
        for k in cols:
            cols[k].append(out[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in cols.items()}


def fit(output, lr, seed, tr_x, tr_h, dv_x, dv_h, te_x, te_rows, provenance):
    run = Path(output) / "ordinal" / f"lr{lr:g}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    metrics_path = run / "metrics.json"
    if metrics_path.exists():
        result = json.loads(metrics_path.read_text())
        assert result["provenance"] == provenance
        assert (run / "best.pt").exists() and (run / "test_scores.csv").exists()
        print(f"REUSE ordinal lr={lr:g} seed={seed}", flush=True)
        return result
    torch.manual_seed(seed)
    model = SESJudge().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    start, best_loss, best_epoch, bad = 1, float("inf"), 0, 0
    history = []
    progress = run / "resume.pt"
    if progress.exists():
        obj = torch.load(progress, map_location="cuda", weights_only=False)
        assert obj["provenance"] == provenance
        model.load_state_dict(obj["model"])
        optimizer.load_state_dict(obj["optimizer"])
        start, best_loss, best_epoch, bad = obj["epoch"]+1, obj["best_loss"], obj["best_epoch"], obj["bad"]
        history = obj["history"]
        # Resume snapshot carries matching best weights; do not mix with later interrupted writes.
        save_torch(run / "best.pt", obj["best_checkpoint"])
        print(f"RESUME ordinal lr={lr:g} seed={seed} epoch={start}", flush=True)
    started = time.monotonic()
    for epoch in range(start, EPOCHS+1):
        if bad >= PATIENCE:
            break
        model.train()
        generator = torch.Generator(device="cuda").manual_seed(seed*10000+epoch)
        order = torch.randperm(len(tr_x), generator=generator, device="cuda")
        total = 0.0
        for i in range(0, len(order), BATCH):
            ix = order[i:i+BATCH]
            optimizer.zero_grad(set_to_none=True)
            loss = ordinal_loss(model(tr_x[ix]), tr_h[ix])
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss: ordinal {lr} {seed} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())*len(ix)
        model.eval()
        with torch.inference_mode():
            dev_loss = float(ordinal_loss(model(dv_x), dv_h))
        if not np.isfinite(dev_loss):
            raise RuntimeError("nonfinite dev loss")
        history.append({"epoch": epoch, "loss": total/len(tr_x), "dev_loss": dev_loss})
        if dev_loss < best_loss:
            best_loss, best_epoch, bad = dev_loss, epoch, 0
            save_torch(run / "best.pt", {
                "model": cpu_state(model), "method": "ordinal", "lr": lr, "seed": seed,
                "epoch": epoch, "dev_loss": dev_loss, "provenance": provenance})
        else:
            bad += 1
        if epoch % 5 == 0 or bad >= PATIENCE or epoch == EPOCHS:
            save_torch(progress, {
                "model": cpu_state(model), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "best_loss": best_loss, "best_epoch": best_epoch, "bad": bad,
                "history": history, "provenance": provenance,
                "best_checkpoint": torch.load(run / "best.pt", map_location="cpu", weights_only=False)})
            atomic_json(run / "history.json", history)
        if epoch % 10 == 0:
            print(f"EPOCH ordinal lr={lr:g} seed={seed} ep={epoch} dev_loss={dev_loss:.5f} best_loss={best_loss:.5f}", flush=True)
    best = torch.load(run / "best.pt", map_location="cuda", weights_only=False)
    assert best["epoch"] == best_epoch
    assert best_epoch == min(history,key=lambda h:h['dev_loss'])['epoch']
    assert best['dev_loss'] == min(h['dev_loss'] for h in history)
    model.load_state_dict(best["model"])
    outputs = evaluate(model, te_x)
    assert all(np.isfinite(v).all() for v in outputs.values())
    # Check the checkpoint's swap symmetry on actual Test features.
    swapped = evaluate(model, te_x[:16, [0, 2, 1]])["score"]
    assert np.allclose(swapped, -outputs["score"][:16], atol=2e-5)
    te_y = np.array([float(r["mean_strength"]) for r in te_rows])
    rho = float(spearmanr(outputs["score"], te_y).statistic)
    assert np.isfinite(rho)
    with (run / "test_scores.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["item", "score", "margin", "sim_a", "sim_b", "mean_strength"])
        for i, r in enumerate(te_rows):
            w.writerow([r["item"]]+[float(outputs[k][i]) for k in ("score", "margin", "sim_a", "sim_b")]+[r["mean_strength"]])
    result = {"method": "ordinal", "lr": lr, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": history[-1]["epoch"], "dev_loss": best_loss,
              "test_spearman": rho, "test_n": len(te_rows), "seconds": time.monotonic()-started,
              "checkpoint": str(run / "best.pt"), "scores": str(run / "test_scores.csv"),
              "provenance": provenance}
    atomic_json(run / "history.json", history)
    atomic_json(metrics_path, result)
    if progress.exists():
        progress.unlink()  # Fully evaluated run retains best.pt and complete history.
    print(f"RESULT ordinal lr={lr:g} seed={seed} epoch={best_epoch} TEST_RHO={rho:.6f}", flush=True)
    return result
