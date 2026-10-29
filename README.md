# Private-LLM — SoloLM

SoloLM is an experimental language-model project built from scratch to learn how transformer-based language models work, with a long-term goal of developing a capable model using free or low-cost tools.

## Current milestone: SoloLM v0

The first version is a small, character-level, decoder-only Transformer trained on the Tiny Shakespeare dataset in Google Colab.

- **Framework:** PyTorch
- **Training environment:** Google Colab (T4 GPU was used for the initial experiment)
- **Model size:** approximately 818,000 trainable parameters
- **Initial training run:** 2,000 steps
- **Artifacts:** training notebook and saved model checkpoint

This is an early research prototype. It can generate Shakespeare-like text, but it is **not yet a general-purpose conversational assistant** and should not be treated as one.

## Repository contents

- `sololm_v0.ipynb` — notebook containing the initial model implementation, training run, and generation example.
- `sololm_v0.pt` — saved checkpoint containing the trained model state and associated configuration/vocabulary data.
- `README.md` — project overview and development notes.

## Run the notebook

1. Open [`sololm_v0.ipynb` in Google Colab](https://colab.research.google.com/github/olaoluwabankole3-cmd/Private-LLM/blob/main/sololm_v0.ipynb).
2. Select a runtime with a GPU if one is available.
3. Review the notebook before running cells. Running the training cells again will start another training run and use additional runtime time.
4. To use the saved checkpoint, make sure the notebook's model definition and configuration match the checkpoint before loading it.

Colab runtimes are temporary. Save important checkpoints outside the runtime, and keep a separate local copy of the checkpoint.

## Development roadmap

1. Make training resumable by saving model weights, optimizer state, training step, and configuration.
2. Add a proper train/validation split and track validation loss.
3. Improve data handling and document dataset provenance and licensing.
4. Add a clean inference script for loading a checkpoint and generating text.
5. Experiment with tokenizer, model architecture, and training settings based on measured results.

## Principles

- Start small and measure progress before scaling up.
- Prefer reproducible experiments and documented datasets.
- Preserve working checkpoints before making major changes.
- Be transparent about current capabilities and limitations.

## Status

**SoloLM v0: initial training experiment completed.** Further work is needed on evaluation, training recovery, data quality, and model capability.
