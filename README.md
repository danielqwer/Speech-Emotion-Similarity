# Toward Human-Aligned Judgement of Speech Emotion Similarity

> [!NOTE]
> **Dataset:** SES-Bench will be made available as soon as data preparation is complete.
>
> **Paper:** The arXiv preprint will be available soon.
>
> **Checkpoint:** Model weights are not included yet.

SES-Judge scores how similar a candidate utterance sounds to a reference in emotional expression. This repository provides training and inference code.

## Installation

Use Python 3.10 and run the following commands from the repository root.

```bash
conda create -n ses-judge python=3.10 -y
conda activate ses-judge
python -m pip install -r requirements.txt
```

The pretrained [WavLM-SER encoder](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes) is downloaded automatically on first use and cached by Hugging Face.

## Inference

Place a trained checkpoint at `checkpoint/ses-judge.pt` before running inference.

```bash
python scripts/infer.py --reference reference.wav --candidate candidate.wav
```

The command returns a cosine similarity score in JSON. Higher scores indicate closer emotional similarity. Multiple audio files can follow `--candidate`; each is scored independently against the reference.

Audio is converted to mono and resampled to 16 kHz automatically. Inference uses CUDA when available and otherwise runs on CPU. Use `--device cpu` to select CPU explicitly.

## Training

Once SES-Bench is available, run:

```bash
python scripts/train.py --dataset ../SES-Bench
```

With the default settings, the trained checkpoint is saved to `runs/SES-Judge/ordinal/lr0.0001_seed4/best.pt`. Pass this path to `--checkpoint` when running inference to use your trained model.
