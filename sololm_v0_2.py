"""SoloLM v0.2: a small GPT-style Transformer with byte-level BPE tokenization.

Free Colab setup:
  !pip -q install tokenizers
  !wget -q https://raw.githubusercontent.com/olaoluwabankole3-cmd/Private-LLM/main/sololm_v0_2.py
  !python sololm_v0_2.py --max-steps 3000

The script downloads WikiText-2 (raw), trains a tokenizer on the training split only,
and writes tokenizer files plus resumable checkpoints to sololm_v0_2_artifacts/.
This is an educational small-model experiment, not a production language model.
"""
import argparse
import math
import os
import urllib.request
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import ByteLevel


DATA_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
SPECIAL_TOKEN = "<|endoftext|>"


def parse_args():
    p = argparse.ArgumentParser(description="Train SoloLM v0.2")
    p.add_argument("--max-steps", type=int, default=3000, help="Total target step count.")
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--output-dir", default="sololm_v0_2_artifacts")
    p.add_argument("--resume", default=None, help="Checkpoint path; tokenizer files must be in output-dir.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def download_dataset(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "wikitext-2-raw-v1.zip"
    extracted = root / "wikitext-2-raw"
    train_path = extracted / "wiki.train.raw"
    valid_path = extracted / "wiki.valid.raw"
    if train_path.exists() and valid_path.exists():
        return train_path, valid_path
    print("Downloading WikiText-2 raw dataset...")
    urllib.request.urlretrieve(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(root)
    unpacked = root / "wikitext-2-raw-v1"
    if unpacked.exists():
        if extracted.exists():
            import shutil
            shutil.rmtree(extracted)
        unpacked.rename(extracted)
    if not train_path.exists() or not valid_path.exists():
        raise RuntimeError("Dataset download completed but expected WikiText-2 files were not found.")
    return train_path, valid_path


def load_or_train_tokenizer(train_path: Path, output_dir: Path, vocab_size: int):
    vocab_path = output_dir / "tokenizer-vocab.json"
    merges_path = output_dir / "tokenizer-merges.txt"
    if vocab_path.exists() and merges_path.exists():
        tokenizer = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
        tokenizer._tokenizer.post_processor = ByteLevel(trim_offsets=False)
        return tokenizer
    print(f"Training byte-level BPE tokenizer (target vocabulary {vocab_size:,})...")
    tokenizer = ByteLevelBPETokenizer()
    def lines():
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                yield line
    tokenizer.train_from_iterator(
        lines(), vocab_size=vocab_size, min_frequency=2,
        special_tokens=[SPECIAL_TOKEN]
    )
    tokenizer._tokenizer.post_processor = ByteLevel(trim_offsets=False)
    tokenizer.save_model(str(output_dir), "tokenizer")
    # save_model creates tokenizer-vocab.json and tokenizer-merges.txt with this prefix.\n    return tokenizer


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return F.dropout(self.proj(out), p=self.dropout, training=self.training)


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class SoloLMv02(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config["block_size"]
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.position_embedding = nn.Embedding(config["block_size"], config["d_model"])
        self.blocks = nn.Sequential(*[
            Block(config["d_model"], config["n_heads"], config["dropout"])
            for _ in range(config["n_layers"])
        ])
        self.final_norm = nn.LayerNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx, targets=None):
        _, length = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(
            torch.arange(length, device=idx.device)
        )
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=200, temperature=0.8, top_k=50):
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits[logits < values[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx


def save_checkpoint(path, model, optimizer, config, step, best_val):
    payload = {
        "format_version": 2, "step": step, "best_val_loss": best_val,
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "config": config, "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = str(path) + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, config, batches):
    model.eval()
    out = {}
    for name, data in (("train", train_data), ("val", val_data)):
        losses = []
        max_start = len(data) - config["block_size"] - 1
        if max_start < 1:
            raise ValueError(f"{name} dataset is too short for block size {config['block_size']}.")
        for _ in range(batches):
            starts = torch.randint(max_start, (config["batch_size"],))
            x = torch.stack([data[i:i + config["block_size"]] for i in starts]).to(device)
            y = torch.stack([data[i + 1:i + config["block_size"] + 1] for i in starts]).to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()
    return out


def main():
    args = parse_args()
    if min(args.max_steps, args.eval_interval, args.eval_batches, args.batch_size, args.block_size) < 1:
        raise ValueError("Training and evaluation settings must be positive.")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    train_path, valid_path = download_dataset(data_dir)
    tokenizer = load_or_train_tokenizer(train_path, output_dir, args.vocab_size)
    vocab_size = tokenizer.get_vocab_size()
    print("Tokenizer vocabulary:", f"{vocab_size:,}")

    train_ids = tokenizer.encode(train_path.read_text(encoding="utf-8")).ids
    val_ids = tokenizer.encode(valid_path.read_text(encoding="utf-8")).ids
    train_data = torch.tensor(train_ids, dtype=torch.long)
    val_data = torch.tensor(val_ids, dtype=torch.long)
    print(f"Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")

    config = {
        "vocab_size": vocab_size, "d_model": args.d_model, "n_heads": args.n_heads,
        "n_layers": args.n_layers, "block_size": args.block_size, "dropout": 0.1,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "tokenizer_type": "byte-level-BPE", "dataset": "WikiText-2 raw"
    }
    model = SoloLMv02(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    start_step, best_val = 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("format_version") != 2:
            raise ValueError("Resume checkpoint is not a SoloLM v0.2 checkpoint.")
        if ckpt["config"] != config:
            raise ValueError("Checkpoint config does not match current settings. Use the original settings to resume.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt["step"])
        best_val = float(ckpt["best_val_loss"])
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"].cpu())
        if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state_all"]])
        print(f"Resumed at step {start_step} from {args.resume}")

    best_path = output_dir / "sololm_v0_2_best.pt"
    last_path = output_dir / "sololm_v0_2_last.pt"
    for step in range(start_step, args.max_steps):
        if step % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, device, config, args.eval_batches)
            print(f"step {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(best_path, model, optimizer, config, step, best_val)
                print("  Saved new best checkpoint:", best_path)
            save_checkpoint(last_path, model, optimizer, config, step, best_val)
        max_start = len(train_data) - config["block_size"] - 1
        starts = torch.randint(max_start, (config["batch_size"],))
        x = torch.stack([train_data[i:i + config["block_size"]] for i in starts]).to(device)
        y = torch.stack([train_data[i + 1:i + config["block_size"] + 1] for i in starts]).to(device)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    save_checkpoint(last_path, model, optimizer, config, args.max_steps, best_val)
    print(f"Training finished at step {args.max_steps}. Last checkpoint: {last_path}")
    prompt = "The future of science is"
    ids = tokenizer.encode(prompt).ids
    context = torch.tensor([ids], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=160)
    print("\n--- SOLOLM v0.2 SAMPLE ---\n")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
