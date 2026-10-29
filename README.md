# Private-LLM — SoloLM

SoloLM is an experimental language-model project built from scratch to learn how transformer-based language models work, with a long-term goal of developing a more capable model using free or low-cost tools.

## Current milestone: SoloLM v0

The first version is a small, character-level, decoder-only Transformer trained on the Tiny Shakespeare dataset in Google Colab.

- **Framework:** PyTorch
- **Training environment:** Google Colab (T4 GPU was used for the initial experiment)
- **Model size:** approximately 818,000 trainable parameters
- **Initial training run:** 2,000 steps
- **Artifacts:** original notebook and trained model checkpoint

This is an early research prototype. It can generate Shakespeare-like text, but it is **not yet a general-purpose conversational assistant**.

## Repository contents

- `sololm_v0.ipynb` — original model implementation, training run, and generation example.
- `sololm_v0.pt` — original trained model weights, configuration, and vocabulary.
- `sololm_v0_1.py` — improved training script with resumable checkpoints, optimizer state, training step, RNG state, and validation-loss tracking.
- `README.md` — project overview and development notes.

## Run SoloLM v0

Open [the original notebook in Google Colab](https://colab.research.google.com/github/olaoluwabankole3-cmd/Private-LLM/blob/main/sololm_v0.ipynb). Avoid running all cells unless you intend to repeat the original training run.

## Try the v0.1 training script in Google Colab

Start a fresh Colab notebook and select a GPU runtime if one is available. If this repository is public, run:

```bash
!wget -q https://raw.githubusercontent.com/olaoluwabankole3-cmd/Private-LLM/main/sololm_v0_1.py
!wget -q https://raw.githubusercontent.com/olaoluwabankole3-cmd/Private-LLM/main/sololm_v0.pt
!python sololm_v0_1.py --init-checkpoint sololm_v0.pt --max-steps 500 --eval-interval 100
```

This initializes the v0.1 run from the original v0 weights. The optimizer starts fresh because the original checkpoint did not store optimizer state. The script writes resumable checkpoints to `sololm_checkpoints/`.

To resume a v0.1 run in the same runtime, use the saved checkpoint and set the **total target step count** higher than the checkpoint's saved step:

```bash
!python sololm_v0_1.py --resume sololm_checkpoints/sololm_v0_1_last.pt --max-steps 1000
```

For a fresh Colab runtime, upload the saved checkpoint folder/file first, or copy the checkpoint to persistent storage. Colab's local files are temporary and can disappear when the runtime is deleted. Download important checkpoints or store them somewhere persistent before disconnecting.

The validation loss is an estimate based on sampled batches from a held-out section of the same Tiny Shakespeare text. It is a useful first diagnostic, not a broad measure of language ability.

## Development roadmap

1. Validate the training and resume workflow in Colab.
2. Add a separate inference script for loading a checkpoint and generating text without training.
3. Improve data handling and document dataset provenance and licensing.
4. Compare training settings using recorded validation results before scaling model size or data.

## Principles

- Start small and measure progress before scaling up.
- Prefer reproducible experiments and documented datasets.
- Preserve working checkpoints before making major changes.
- Be transparent about current capabilities and limitations.

## Status

**SoloLM v0:** initial training experiment completed and checkpoint preserved.  
**SoloLM v0.1:** resumable training script added; it still needs to be run and validated in the intended environment.
