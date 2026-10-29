#!/usr/bin/env python3
"""Generate text with a trained SoloLM v0.2 checkpoint.

Example:
  python sololm_v0_2_generate.py --prompt "The future of science is"
  python sololm_v0_2_generate.py --checkpoint sololm_v0_2_artifacts/sololm_v0_2_last.pt --prompt "Artificial intelligence"
"""
import argparse
from pathlib import Path

import torch
from tokenizers import ByteLevelBPETokenizer

from sololm_v0_2 import SoloLMv02


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with SoloLM v0.2")
    parser.add_argument(
        "--checkpoint",
        default="sololm_v0_2_artifacts/sololm_v0_2_best.pt",
        help="Path to a SoloLM v0.2 checkpoint (best checkpoint by default).",
    )
    parser.add_argument(
        "--output-dir",
        default="sololm_v0_2_artifacts",
        help="Directory containing tokenizer-vocab.json and tokenizer-merges.txt.",
    )
    parser.add_argument("--prompt", default="The future of science is")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than 0.")
    if args.top_k < 0:
        raise ValueError("--top-k must be 0 (disabled) or greater.")

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    vocab_path = output_dir / "tokenizer-vocab.json"
    merges_path = output_dir / "tokenizer-merges.txt"
    for path in (checkpoint_path, vocab_path, merges_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file not found: {path}. Make sure the checkpoint "
                "and matching tokenizer files are present."
            )

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("format_version") != 2 or "model_state_dict" not in checkpoint:
        raise ValueError("This is not a supported SoloLM v0.2 checkpoint.")
    config = checkpoint["config"]

    tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    model = SoloLMv02(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt).ids
    if not prompt_ids:
        raise ValueError("The prompt produced no tokens. Please provide a non-empty prompt.")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    top_k = args.top_k if args.top_k > 0 else None

    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
        )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"Best validation loss recorded: {checkpoint.get('best_val_loss', 'unknown')}")
    print("\n--- SOLOLM v0.2 GENERATED TEXT ---\n")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
