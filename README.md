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
- `sololm_v0_1.py` — training script with resumable checkpoints, optimizer state, training step, RNG state, and validation-loss tracking.
- `sololm_v0_1_best.pt` — best saved v0.1 checkpoint by observed validation loss.
- `sololm_v0_1_last.pt` — latest v0.1 checkpoint, including optimizer state for resuming training.
- `sololm_generate.py` — standalone text-generation script that loads a saved checkpoint.
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

## Generate text without training

In a fresh Google Colab notebook, select a GPU runtime if available and download the inference script, model definition, and best checkpoint:

```bash
!wget -q https://raw.githubusercontent.com/olaoluwabankole3-cmd/Private-LLM/main/sololm_generate.py
!wget -q https://raw.githubusercontent.com/olaoluwabankole3-cmd/Private-LLM/main/sololm_v0_1.py
!wget -q https://raw.githubusercontent.com/olaoluwabankole3-cmd/Private-LLM/main/sololm_v0_1_best.pt
!python sololm_generate.py --checkpoint sololm_v0_1_best.pt --prompt "ROMEO:" --max-new-tokens 400
```

You can change the prompt and generation length. The model is character-level, so prompt characters must exist in its training vocabulary. Generation does not train or modify the checkpoint.

## SoloLM v0.2: subword-tokenized model

v0.2 is a separate experiment. It does **not** overwrite the v0/v0.1 code or checkpoints.

- **Architecture:** small decoder-only Transformer with causal self-attention.
- **Tokenizer:** byte-level Byte Pair Encoding (BPE), which represents text using reusable byte/subword pieces rather than one token per character.
- **Dataset:** WikiText-2 raw, with the official train and validation splits kept separate.
- **Training:** PyTorch; free Google Colab GPU recommended.
- **Artifacts:** tokenizer files and resumable best/last checkpoints are written under `sololm_v0_2_artifacts/`.

### Run v0.2 in Google Colab

Open a new Colab notebook, choose a GPU runtime if available, then run these cells in order:

```bash
!pip -q install tokenizers
!wget -q -O sololm_v0_2.py https://raw.githubusercontent.com/olaoluwabankole3-cmd/Private-LLM/main/sololm_v0_2.py
!python sololm_v0_2.py --max-steps 3000 --eval-interval 200 --eval-batches 10
```

The first run downloads WikiText-2 and trains the BPE tokenizer. This can take a little while. Training then reports train and validation loss and writes:
- `sololm_v0_2_artifacts/sololm_v0_2_best.pt`
- `sololm_v0_2_artifacts/sololm_v0_2_last.pt`
- `sololm_v0_2_artifacts/tokenizer-vocab.json`
- `sololm_v0_2_artifacts/tokenizer-merges.txt`

Download the entire `sololm_v0_2_artifacts` folder before the Colab runtime ends. Colab's local files are temporary. Keep the v0.2 tokenizer files alongside its checkpoint; the checkpoint's token IDs only make sense with the tokenizer trained for that run.

To continue training in the same runtime, use the last checkpoint and the same model settings:

```bash
!python sololm_v0_2.py --resume sololm_v0_2_artifacts/sololm_v0_2_last.pt --max-steps 5000
```

Here, `--max-steps 5000` means a total target of 5,000 steps, not 5,000 additional steps. For a fresh runtime, upload/restore the artifact folder first and keep the tokenizer files in the same output directory. Do not change architecture settings when resuming a checkpoint.

**Important:** validation loss is only comparable within the same dataset/tokenizer setup. It cannot be directly compared numerically with v0.1's character-level loss. v0.2 is still a small research model and is not a general-purpose assistant.

## Development roadmap

1. Test standalone generation from the committed checkpoint.
2. Verify resume training from `sololm_v0_1_last.pt` and record comparable validation results.
3. Improve data handling and document dataset provenance and licensing.
4. Compare training settings using recorded validation results before scaling model size or data.

## Principles

- Start small and measure progress before scaling up.
- Prefer reproducible experiments and documented datasets.
- Preserve working checkpoints before making major changes.
- Be transparent about current capabilities and limitations.

## Status

**SoloLM v0:** initial training experiment completed and checkpoint preserved.  
**SoloLM v0.1:** 500-step continuation completed on a CUDA runtime. The observed validation loss improved from approximately 2.2003 at the initial evaluation to 2.1591 at step 400. Text generation runs, but output is still mostly incoherent, as expected for this early character-level prototype. Standalone generation and resume training should be tested again from the committed files.
