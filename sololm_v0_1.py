"""SoloLM v0.1: resumable character-level Transformer training.

Example in Google Colab:
  !python sololm_v0_1.py --init-checkpoint sololm_v0.pt --max-steps 500

Resume a v0.1 run:
  !python sololm_v0_1.py --resume sololm_v0_1_last.pt --max-steps 1500

The --max-steps value is the total target step count, not additional steps.
"""

import argparse
import math
import os
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/"
    "char-rnn/master/data/tinyshakespeare/input.txt"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SoloLM v0.1")
    parser.add_argument("--max-steps", type=int, default=2000,
                        help="Total target training steps.")
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--init-checkpoint", type=str, default=None,
                        help="Initialize model weights from the original v0 checkpoint.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a v0.1 checkpoint including optimizer state.")
    parser.add_argument("--output-dir", type=str, default="sololm_checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_text():
    url = DATA_URL
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(
            "Could not download Tiny Shakespeare. Check internet access and retry."
        ) from exc


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size))
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        head_dim = channels // self.n_heads
        q = q.view(batch, length, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = scores.masked_fill(self.mask[:length, :length] == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class SoloLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        n_layers = config["n_layers"]
        block_size = config["block_size"]
        dropout = config["dropout"]
        vocab_size = config["vocab_size"]
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            Block(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        _, length = idx.shape
        positions = torch.arange(length, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=300, temperature=0.8):
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def save_checkpoint(path, model, optimizer, config, stoi, itos, step, best_val):
    checkpoint = {
        "format_version": 1,
        "step": step,
        "best_val_loss": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "stoi": stoi,
        "itos": itos,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = str(path) + ".tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, config, eval_batches):
    model.eval()
    results = {}
    for name, source in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_batches):
            starts = torch.randint(len(source) - config["block_size"] - 1,
                                   (config["batch_size"],))
            x = torch.stack([source[i:i + config["block_size"]] for i in starts])
            y = torch.stack([source[i + 1:i + config["block_size"] + 1] for i in starts])
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def main():
    args = parse_args()
    if args.max_steps < 1 or args.eval_interval < 1 or args.eval_batches < 1:
        raise ValueError("Step and evaluation settings must be positive.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    text = load_text()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]

    config = {
        "vocab_size": len(chars), "d_model": 128, "n_heads": 4,
        "n_layers": 4, "block_size": 128, "dropout": 0.1,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
    }
    print(f"Characters: {len(text):,} | Vocabulary: {len(chars)}")
    print(f"Train/validation characters: {len(train_data):,}/{len(val_data):,}")

    model = SoloLM(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    start_step = 0
    best_val = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("This is not a resumable SoloLM v0.1 checkpoint.")
        if checkpoint["config"]["vocab_size"] != config["vocab_size"]:
            raise ValueError("Checkpoint vocabulary does not match the current dataset.")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint["step"])
        best_val = float(checkpoint.get("best_val_loss", float("inf")))
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        stoi, itos = checkpoint["stoi"], checkpoint["itos"]
        print(f"Resumed at step {start_step} from {args.resume}")
    elif args.init_checkpoint:
        old = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if old["config"]["vocab_size"] != config["vocab_size"]:
            raise ValueError("Original checkpoint vocabulary differs from current dataset.")
        model.load_state_dict(old["model_state_dict"])
        print(f"Initialized weights from {args.init_checkpoint}; optimizer starts fresh.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "sololm_v0_1_last.pt"
    best_path = output_dir / "sololm_v0_1_best.pt"

    model.train()
    for step in range(start_step, args.max_steps):
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, device, config, args.eval_batches)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(best_path, model, optimizer, config, stoi, itos, step, best_val)
                print(f"  New best validation checkpoint: {best_path}")
            save_checkpoint(last_path, model, optimizer, config, stoi, itos, step, best_val)

        starts = torch.randint(len(train_data) - config["block_size"] - 1,
                               (config["batch_size"],))
        x = torch.stack([train_data[i:i + config["block_size"]] for i in starts]).to(device)
        y = torch.stack([train_data[i + 1:i + config["block_size"] + 1] for i in starts]).to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    final_step = args.max_steps
    save_checkpoint(last_path, model, optimizer, config, stoi, itos, final_step, best_val)
    print(f"Training finished at step {final_step}. Checkpoint: {last_path}")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start, max_new_tokens=400)
    sample = "".join(itos[int(i)] for i in generated[0].tolist())
    print("\n--- SOLOLM SAMPLE ---\n")
    print(sample)


if __name__ == "__main__":
    main()
