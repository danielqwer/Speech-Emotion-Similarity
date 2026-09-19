"""Train SES-Judge with its final ordinal objective and representation mapping."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, default=ROOT.parent / 'SES-Bench')
    p.add_argument('--encoder-dir', type=Path, help='Local WavLM-SER directory; default downloads the pinned encoder to the Hugging Face cache')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/SES-Judge')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=4)
    a = p.parse_args()
    for split in ('Train', 'Test'):
        source = a.dataset / 'statistic' / f'{split}.csv'
        if not source.is_file():
            p.error(f'Missing {source}. Training requires SES-Bench audio and annotations; the dataset is not bundled.')
    import torch
    import training
    from utils import read_rows, speaker_split, atomic_json, digest
    from infer import load_encoder, features
    if not torch.cuda.is_available():
        p.error('Training requires a CUDA GPU')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows, test = read_rows('Train', a.dataset), read_rows('Test', a.dataset)
    train_indices, dev_indices = speaker_split(rows)
    keys = sorted({row[key]
                   for records in (rows, test)
                   for row in records for key in ('path_ref', 'path_a', 'path_b')})
    encoder = load_encoder('cuda', a.encoder_dir)
    # Features are held in memory for this run, not saved as embedding caches.
    bank = []
    for i, key in enumerate(keys):
        bank.append(features(a.dataset / key, encoder).cpu())
        if i % 500 == 0:
            print('EXTRACT', i, len(keys), flush=True)
    del encoder
    torch.cuda.empty_cache()
    vectors = torch.stack(bank).cuda()
    del bank
    positions = {key: i for i, key in enumerate(keys)}
    x, histogram = training.tensors(rows, positions, vectors)
    test_x, _ = training.tensors(test, positions, vectors)
    provenance = dict(
        architecture='25 per-layer projections, normalization, learned sum, final projection',
        train_sha256=digest(a.dataset / 'statistic/Train.csv'), test_sha256=digest(a.dataset / 'statistic/Test.csv'),
        epochs=training.EPOCHS, patience=training.PATIENCE, batch=training.BATCH,
        weight_decay=training.WEIGHT_DECAY, split_seed=42,
        train_n=len(train_indices), dev_n=len(dev_indices),
        epoch_selection='minimum held-out ordinal loss')
    atomic_json(a.output / 'split.json', dict(
        train_items=[rows[i]['item'] for i in train_indices],
        dev_items=[rows[i]['item'] for i in dev_indices], split_seed=42))
    training.fit(a.output, a.lr, a.seed, x[train_indices], histogram[train_indices],
                 x[dev_indices], histogram[dev_indices], test_x, test, provenance)


if __name__ == '__main__':
    main()
