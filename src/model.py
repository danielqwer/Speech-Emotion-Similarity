import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import WavLMConfig, WavLMModel


class FrozenSER(nn.Module):
    """Single-utterance feature extractor; accepts mono audio sampled at 16 kHz."""

    def __init__(self, directory=None):
        super().__init__()
        from huggingface_hub import hf_hub_download, snapshot_download

        ser_files = ('config.json', 'pytorch_model.bin')
        if directory is None:
            directory = Path(snapshot_download('3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes',
                             revision='00d0e12ba9bf957f5aeea36e8663c8c61cb50ac9',
                             allow_patterns=ser_files))
            config_path = hf_hub_download('microsoft/wavlm-large', 'config.json',
                                         revision='c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c')
        else:
            directory = Path(directory).expanduser()
            missing = [name for name in ser_files if not (directory / name).is_file()]
            if missing:
                snapshot_download('3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes',
                                  revision='00d0e12ba9bf957f5aeea36e8663c8c61cb50ac9',
                                  allow_patterns=missing, local_dir=directory)
            config_path = directory / 'wavlm_config/config.json'
            if not config_path.is_file():
                hf_hub_download('microsoft/wavlm-large', 'config.json',
                                revision='c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c',
                                local_dir=directory / 'wavlm_config')
        metadata = json.loads((directory / 'config.json').read_text())
        config = WavLMConfig.from_json_file(config_path)
        if (config.num_hidden_layers, config.hidden_size, metadata['classifier_hidden_layers']) != (24, 1024, 1):
            raise ValueError('Expected the 24-layer, 1024-dimensional Odyssey WavLM-SER checkpoint')
        self.mean = float(metadata['mean'])
        self.std = float(metadata['std'])
        self.wavlm = WavLMModel(config)
        self.attention = nn.Linear(1024, 1024)
        self.register_buffer('attention_vector', torch.empty(1024, 1))
        self.hidden = nn.Sequential(nn.Linear(2048, 1024), nn.LayerNorm(1024), nn.ReLU())

        weights = torch.load(directory / 'pytorch_model.bin', map_location='cpu', weights_only=True)
        # Map stored tensor names onto our inference modules, checking every tensor used.
        backbone = {key.removeprefix('ssl_model.'): weights.pop(key)
                    for key in list(weights) if key.startswith('ssl_model.')}
        self.wavlm.load_state_dict(backbone, strict=True)
        mapping = {
            'attention.weight': 'pool_model.sap_linear.weight',
            'attention.bias': 'pool_model.sap_linear.bias',
            'attention_vector': 'pool_model.attention',
            'hidden.0.weight': 'ser_model.fc.0.0.weight',
            'hidden.0.bias': 'ser_model.fc.0.0.bias',
            'hidden.1.weight': 'ser_model.fc.0.1.weight',
            'hidden.1.bias': 'ser_model.fc.0.1.bias',
        }
        for target, source in mapping.items():
            destination = self.get_buffer(target) if target == 'attention_vector' else self.get_parameter(target)
            value = weights.pop(source)
            if destination.shape != value.shape:
                raise ValueError(f'Unexpected checkpoint shape for {source}: {tuple(value.shape)}')
            with torch.no_grad():
                destination.copy_(value)
        # The VAD prediction layer is not used to construct emotion embeddings.
        if set(weights) != {'ser_model.out.0.weight', 'ser_model.out.0.bias'}:
            raise ValueError(f'Unexpected remaining checkpoint tensors: {sorted(weights)}')
        self.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, waveform):
        if waveform.ndim != 1:
            raise ValueError('Expected one unpadded mono waveform')
        standardized = (waveform[None] - self.mean) / (self.std + 1e-6)
        output = self.wavlm(standardized, attention_mask=torch.ones_like(standardized, dtype=torch.long),
                            output_hidden_states=True)
        frames = output.last_hidden_state
        attention = (self.attention(frames).tanh() @ self.attention_vector).softmax(dim=1)
        mean = (frames * attention).sum(dim=1)
        variance = (frames.square() * attention).sum(dim=1) - mean.square()
        hidden = self.hidden(torch.cat((mean, variance.clamp_min(1e-5).sqrt()), dim=-1))
        layers = [state.mean(dim=1).squeeze(0) for state in output.hidden_states[1:]]
        return torch.stack([*layers, hidden.squeeze(0)])


class SESJudge(nn.Module):
    def __init__(self):
        super().__init__()
        # Preserve parameter names and initialization for the released checkpoint.
        self.projection = nn.Linear(1024, 1024, bias=False)
        nn.init.eye_(self.projection.weight)
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.threshold_steps = nn.Parameter(torch.full((3,), math.log(math.expm1(1.0))))
        self.layer_logits = nn.Parameter(torch.zeros(25))
        self.source_projections = nn.Parameter(torch.eye(1024).repeat(25, 1, 1))

    def embed(self, features):
        """Map (..., 25, 1024) encoder features to unit-length embeddings."""
        assert features.shape[-2:] == (25, 1024)
        projected = torch.einsum('...ld,ldh->...lh', features, self.source_projections)
        projected = F.normalize(projected, dim=-1)
        mixed = (projected * self.layer_logits.softmax(0)[:, None]).sum(-2)
        return F.normalize(self.projection(mixed), dim=-1)

    def forward(self, features):
        """Score (reference, A, B) triplets and predict seven ordered responses."""
        z = self.embed(features)
        sim_a = (z[:, 0] * z[:, 1]).sum(-1)
        sim_b = (z[:, 0] * z[:, 2]).sum(-1)
        margin = sim_a - sim_b
        eta = self.log_scale.clamp(-5, 6).exp() * margin
        positive = torch.cumsum(F.softplus(self.threshold_steps) + 1e-4, 0)
        thresholds = torch.cat([-positive.flip(0), positive])
        v = thresholds[None, :] - eta[:, None]
        # Stable log differences of adjacent cumulative probabilities.
        interior = (v[:, 1:] + torch.log(-torch.expm1(v[:, :-1] - v[:, 1:]))
                    - F.softplus(v[:, 1:]) - F.softplus(v[:, :-1]))
        logp = torch.cat([F.logsigmoid(v[:, :1]), interior, F.logsigmoid(-v[:, -1:])], -1)
        return dict(sim_a=sim_a, sim_b=sim_b, margin=margin, score=margin, logp=logp)


def ordinal_loss(output, histogram):
    """Cross-entropy against annotator proportions ordered B3 through A3."""
    return -(histogram * output['logp']).sum(-1).mean()
