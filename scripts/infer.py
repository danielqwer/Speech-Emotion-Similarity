"""Score a reference and one or more candidates using SES-Judge."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load_encoder(device, model_dir=None):
    from model import FrozenSER
    return FrozenSER(model_dir).to(device)


def features(path, encoder):
    import torch
    from utils import audio
    device = next(encoder.parameters()).device
    return encoder(torch.from_numpy(audio(path)).to(device))


def load_judge(checkpoint, device):
    import torch
    from model import SESJudge
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = SESJudge()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, nargs='+', required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoint/ses-judge.pt')
    p.add_argument('--encoder-dir', type=Path, help='Local WavLM-SER directory; default downloads the pinned encoder to the Hugging Face cache')
    p.add_argument('--device', default='auto', help='auto selects CUDA when available, otherwise CPU')
    a = p.parse_args()
    import torch
    if a.device == 'auto':
        a.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for path in [a.reference, *a.candidate, a.checkpoint]:
        if not path.is_file():
            p.error(f'File not found: {path}')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = load_encoder(a.device, a.encoder_dir)
    model = load_judge(a.checkpoint, a.device)
    with torch.inference_mode():
        reference = model.embed(features(a.reference, encoder))
        scores = []
        for path in a.candidate:
            similarity = float((reference * model.embed(features(path, encoder))).sum())
            scores.append(dict(candidate=str(path), similarity=similarity))
    print(json.dumps(dict(reference=str(a.reference), scores=scores), indent=2))


if __name__ == '__main__':
    main()
