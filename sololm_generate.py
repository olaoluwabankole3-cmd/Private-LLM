"""Generate text with a saved SoloLM v0.1 checkpoint.

Example:
  python sololm_generate.py --checkpoint sololm_v0_1_best.pt --prompt "ROMEO:" --max-new-tokens 400
"""
import argparse
from pathlib import Path

import torch

from sololm_v0_1 import SoloLM


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with SoloLM")
    parser.add_argument("--checkpoint", default="sololm_v0_1_best.pt",
                        help="Path to a SoloLM v0.1 checkpoint.")
    parser.add_argument("--prompt", default="ROMEO:",
                        help="Starting text. Characters must exist in the checkpoint vocabulary.")
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Lower is more conservative; higher is more random.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Download it from the repository "
            "or upload it to the current Colab runtime."
        )

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]
    unknown = sorted(set(args.prompt) - set(stoi))
    if unknown:
        printable = ", ".join(repr(char) for char in unknown)
        raise ValueError(
            f"Prompt contains characters missing from this checkpoint vocabulary: {printable}. "
            "Try a simpler prompt using characters from the training text."
        )

    model = SoloLM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_ids = torch.tensor(
        [[stoi[char] for char in args.prompt]],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    generated = "".join(itos[int(token)] for token in output_ids[0].tolist())
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training step recorded: {checkpoint.get('step', 'unknown')}")
    print("\n--- SOLOLM OUTPUT ---\n")
    print(generated)


if __name__ == "__main__":
    main()
