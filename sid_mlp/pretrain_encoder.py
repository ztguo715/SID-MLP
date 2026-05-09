"""SID-MLP++ encoder distillation.

SID-MLP++ replaces the frozen TIGER encoder with a position-specific MLP
encoder. Stage 1 matches frozen T5 encoder states with MSE. Stage 2 trains the
same SID-MLP decoder on hidden states produced by this encoder.
"""

import argparse
import os
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DEFAULT_D_MODEL = 128
DEFAULT_FFN_DIM = 2048
DEFAULT_DROPOUT = 0.1
DEFAULT_NUM_LAYERS = 4


class PairedSeqDataset(Dataset):
    """Pairs raw token embeddings with teacher encoder outputs."""

    def __init__(self, raw_dir, teacher_dir):
        raw_data = torch.load(
            os.path.join(raw_dir, "encoder_sequences.pt"),
            map_location="cpu",
            weights_only=False,
        )
        teacher_data = torch.load(
            os.path.join(teacher_dir, "encoder_sequences.pt"),
            map_location="cpu",
            weights_only=False,
        )

        self.seqs = []
        for idx in tqdm(sorted(set(raw_data.keys()) & set(teacher_data.keys())), desc="Pairing"):
            raws = raw_data[idx]
            teachers = teacher_data[idx]
            for ctx_idx in range(min(len(raws), len(teachers))):
                raw = raws[ctx_idx]
                teacher = teachers[ctx_idx]
                if raw.shape == teacher.shape:
                    self.seqs.append((raw, teacher))

        if not self.seqs:
            raise RuntimeError("No paired raw/teacher encoder sequences found.")
        self.max_len = max(seq[0].shape[0] for seq in self.seqs)
        self.d_model = self.seqs[0][0].shape[1]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        raw, teacher = self.seqs[idx]
        seq_len = raw.shape[0]
        raw_pad = torch.zeros(self.max_len, self.d_model)
        teacher_pad = torch.zeros(self.max_len, self.d_model)
        mask = torch.zeros(self.max_len, dtype=torch.bool)
        raw_pad[:seq_len] = raw
        teacher_pad[:seq_len] = teacher
        mask[:seq_len] = True
        return raw_pad, teacher_pad, mask


class GpuPairedSeqDataset(Dataset):
    """GPU-resident paired dataset; __getitem__ returns only the row index."""

    def __init__(self, raw_gpu: torch.Tensor, teacher_gpu: torch.Tensor, mask_gpu: torch.Tensor):
        self.raw_gpu = raw_gpu
        self.teacher_gpu = teacher_gpu
        self.mask_gpu = mask_gpu

    def __len__(self):
        return self.raw_gpu.shape[0]

    def __getitem__(self, idx):
        return idx


def collate_fn_gpu(dataset: GpuPairedSeqDataset, indices):
    idx = torch.tensor(indices, dtype=torch.long, device=dataset.raw_gpu.device)
    return dataset.raw_gpu[idx].float(), dataset.teacher_gpu[idx].float(), dataset.mask_gpu[idx]


def build_gpu_resident_dataset(raw_dir, teacher_dir, device, storage_dtype=torch.float16):
    """Load paired sequences into padded tensors on the selected device."""
    raw_data = torch.load(
        os.path.join(raw_dir, "encoder_sequences.pt"),
        map_location="cpu",
        weights_only=False,
    )
    teacher_data = torch.load(
        os.path.join(teacher_dir, "encoder_sequences.pt"),
        map_location="cpu",
        weights_only=False,
    )

    pairs = []
    max_len = 0
    d_model = None
    for idx in tqdm(sorted(set(raw_data.keys()) & set(teacher_data.keys())), desc="Pairing", ncols=100):
        raws = raw_data[idx]
        teachers = teacher_data[idx]
        for ctx_idx in range(min(len(raws), len(teachers))):
            raw = raws[ctx_idx]
            teacher = teachers[ctx_idx]
            if raw.shape != teacher.shape:
                continue
            d_model = raw.shape[1] if d_model is None else d_model
            max_len = max(max_len, raw.shape[0])
            pairs.append((idx, ctx_idx, raw.shape[0]))

    if not pairs:
        raise RuntimeError("No paired raw/teacher encoder sequences found.")

    n_pairs = len(pairs)
    raw_gpu = torch.zeros(n_pairs, max_len, d_model, dtype=storage_dtype, device=device)
    teacher_gpu = torch.zeros(n_pairs, max_len, d_model, dtype=storage_dtype, device=device)
    mask_gpu = torch.zeros(n_pairs, max_len, dtype=torch.bool, device=device)

    for row, (idx, ctx_idx, seq_len) in enumerate(tqdm(pairs, desc="Copying", ncols=100)):
        raw_gpu[row, :seq_len] = raw_data[idx][ctx_idx].to(device=device, dtype=storage_dtype)
        teacher_gpu[row, :seq_len] = teacher_data[idx][ctx_idx].to(device=device, dtype=storage_dtype)
        mask_gpu[row, :seq_len] = True

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return GpuPairedSeqDataset(raw_gpu, teacher_gpu, mask_gpu)


class SIDMLPPlusPlusEncoder(nn.Module):
    """Position-specific MLP encoder used by SID-MLP++.

    The serialized TIGER input follows a repeating digit layout:
    ``[user, d1, d2, d3, d4, ..., eos, pad...]``. Each layer computes a global
    mean context over valid tokens and applies one of four MLPs according to the
    token's digit position. Special tokens share the fourth MLP.
    """

    def __init__(
        self,
        d_model=DEFAULT_D_MODEL,
        ffn_dim=DEFAULT_FFN_DIM,
        dropout=DEFAULT_DROPOUT,
        num_layers=DEFAULT_NUM_LAYERS,
        max_seq_len=256,
        num_positions=4,
    ):
        super().__init__()
        self.num_positions = num_positions
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            ffns = nn.ModuleList()
            for _ in range(num_positions):
                ffns.append(nn.Sequential(
                    nn.Linear(d_model + d_model, ffn_dim),
                    nn.LayerNorm(ffn_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, d_model),
                ))
            self.layers.append(ffns)
        self.norm = nn.LayerNorm(d_model)

    def _position_ids(self, valid, seq_len):
        batch_size = valid.shape[0]
        positions = torch.arange(seq_len, device=valid.device).unsqueeze(0).expand(batch_size, -1)
        pos_ids = ((positions - 1) % self.num_positions).long()
        pos_ids[:, 0] = self.num_positions - 1

        valid_lens = valid.squeeze(-1).sum(dim=1).long()
        eos_pos = (valid_lens - 1).clamp(0, seq_len - 1)
        pos_ids[torch.arange(batch_size, device=valid.device), eos_pos] = self.num_positions - 1
        return pos_ids

    def forward(self, x, src_key_padding_mask=None):
        batch_size, seq_len, _d_model = x.shape
        if src_key_padding_mask is not None:
            valid = (~src_key_padding_mask).unsqueeze(-1).to(x.dtype)
        else:
            valid = torch.ones(batch_size, seq_len, 1, device=x.device, dtype=x.dtype)

        if self.pos_embed is not None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = x + self.pos_embed(positions)

        pos_ids = self._position_ids(valid, seq_len)
        for ffns in self.layers:
            denom = valid.sum(dim=1, keepdim=True).clamp_min(1)
            global_ctx = (x * valid).sum(dim=1, keepdim=True) / denom
            cat_input = torch.cat([x, global_ctx.expand_as(x)], dim=-1)
            update = torch.zeros_like(x)
            for pos in range(self.num_positions):
                update = update + ffns[pos](cat_input) * (pos_ids == pos).unsqueeze(-1)
            x = x + update

        return self.norm(x)


def build_encoder(
    d_model=DEFAULT_D_MODEL,
    ffn_dim=DEFAULT_FFN_DIM,
    dropout=DEFAULT_DROPOUT,
    num_layers=DEFAULT_NUM_LAYERS,
    max_seq_len=256,
):
    return SIDMLPPlusPlusEncoder(
        d_model=d_model,
        ffn_dim=ffn_dim,
        dropout=dropout,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )


def infer_encoder_config(state, defaults=None):
    defaults = dict(defaults or {})
    layer_ids = {
        int(key.split(".")[1])
        for key in state
        if key.startswith("layers.") and key.split(".")[1].isdigit()
    }
    first_ffn = state.get("layers.0.0.0.weight")
    norm_weight = state.get("norm.weight")
    pos_weight = state.get("pos_embed.weight")

    config = {
        "d_model": int(defaults.get("d_model", DEFAULT_D_MODEL)),
        "ffn_dim": int(defaults.get("ffn_dim", DEFAULT_FFN_DIM)),
        "dropout": float(defaults.get("dropout", DEFAULT_DROPOUT)),
        "num_layers": int(defaults.get("num_layers", DEFAULT_NUM_LAYERS)),
        "max_seq_len": int(defaults.get("max_seq_len", 256)),
    }
    if norm_weight is not None:
        config["d_model"] = int(norm_weight.shape[0])
    elif first_ffn is not None:
        config["d_model"] = int(first_ffn.shape[1] // 2)
    if first_ffn is not None:
        config["ffn_dim"] = int(first_ffn.shape[0])
    if layer_ids:
        config["num_layers"] = max(layer_ids) + 1
    if pos_weight is not None:
        config["max_seq_len"] = int(pos_weight.shape[0])
    else:
        raise RuntimeError("SID-MLP++ encoder checkpoints must include pos_embed.weight.")
    return config


def mse_on_valid_tokens(pred, teacher, mask):
    diff = (pred - teacher) ** 2 * mask.unsqueeze(-1).float()
    return diff.sum() / (mask.sum().clamp_min(1) * pred.shape[-1])


@torch.no_grad()
def evaluate_mse(encoder, loader, device):
    encoder.eval()
    total = 0.0
    count = 0
    for raw, teacher, mask in loader:
        raw = raw.to(device)
        teacher = teacher.to(device)
        mask = mask.to(device)
        pred = encoder(raw, src_key_padding_mask=~mask)
        total += float(mse_on_valid_tokens(pred, teacher, mask).item())
        count += 1
    return total / max(count, 1)


def train_encoder(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.gpu_resident:
        storage_dtype = torch.float16 if args.storage_fp16 else torch.float32
        dataset = build_gpu_resident_dataset(
            args.raw_dir,
            args.teacher_dir,
            device,
            storage_dtype=storage_dtype,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda indices: collate_fn_gpu(dataset, indices),
        )
    else:
        dataset = PairedSeqDataset(args.raw_dir, args.teacher_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    val_loader = None
    if args.val_raw_dir and args.val_teacher_dir:
        val_dataset = PairedSeqDataset(args.val_raw_dir, args.val_teacher_dir)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    encoder = build_encoder(
        d_model=args.d_model,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
    ).to(device)
    print(
        f"SID-MLP++ encoder: d_model={args.d_model} ffn_dim={args.ffn_dim} "
        f"layers={args.num_layers} dropout={args.dropout} pos_embed=True",
        flush=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        total = 0.0
        count = 0
        for raw, teacher, mask in tqdm(loader, desc=f"SID-MLP++ stage1 {epoch}/{args.epochs}", ncols=100):
            raw = raw.to(device)
            teacher = teacher.to(device)
            mask = mask.to(device)
            pred = encoder(raw, src_key_padding_mask=~mask)
            loss = mse_on_valid_tokens(pred, teacher, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.item())
            count += 1

        scheduler.step()
        train_loss = total / max(count, 1)
        val_loss = evaluate_mse(encoder, val_loader, device) if val_loader is not None else train_loss

        print(
            f"Epoch {epoch}: train_mse={train_loss:.6f} "
            f"val_mse={val_loss:.6f}",
            flush=True,
        )

    torch.save(encoder.state_dict(), args.output)
    print(f"Saved final SID-MLP++ encoder: {args.output}", flush=True)


def extract_transformed(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.encoder_ckpt, map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    config = infer_encoder_config(state)
    encoder = build_encoder(**config).to(device)
    encoder.load_state_dict(state, strict=True)
    encoder.eval()

    raw_data = torch.load(
        os.path.join(args.raw_dir, "encoder_sequences.pt"),
        map_location="cpu",
        weights_only=False,
    )
    transformed = {}
    with torch.no_grad():
        for idx in tqdm(sorted(raw_data.keys()), desc="SID-MLP++ transform", ncols=100):
            out_list = []
            for ctx in raw_data[idx]:
                hidden = encoder(ctx.unsqueeze(0).to(device)).squeeze(0).cpu()
                out_list.append(hidden)
            transformed[idx] = out_list

    os.makedirs(args.extract_out, exist_ok=True)
    torch.save(transformed, os.path.join(args.extract_out, "encoder_sequences.pt"))
    src_mapping = os.path.join(args.raw_dir, "item_mapping.json")
    if os.path.exists(src_mapping):
        shutil.copy2(src_mapping, os.path.join(args.extract_out, "item_mapping.json"))


def add_encoder_args(parser):
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--teacher_dir", type=str, default=None)
    parser.add_argument("--val_raw_dir", type=str, default=None)
    parser.add_argument("--val_teacher_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--extract_out", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--ffn_dim", type=int, default=DEFAULT_FFN_DIM)
    parser.add_argument("--num_layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gpu_resident", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--storage_fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)


def main(args=None):
    parser = argparse.ArgumentParser(description="Train or apply the SID-MLP++ encoder")
    add_encoder_args(parser)
    parsed = parser.parse_args(args)
    if parsed.extract:
        if not parsed.encoder_ckpt or not parsed.extract_out:
            raise SystemExit("--encoder_ckpt and --extract_out are required with --extract")
        extract_transformed(parsed)
    else:
        if not parsed.teacher_dir or not parsed.output:
            raise SystemExit("--teacher_dir and --output are required for stage1 training")
        train_encoder(parsed)


if __name__ == "__main__":
    main()


__all__ = [
    "GpuPairedSeqDataset",
    "PairedSeqDataset",
    "SIDMLPPlusPlusEncoder",
    "add_encoder_args",
    "build_encoder",
    "build_gpu_resident_dataset",
    "collate_fn_gpu",
    "extract_transformed",
    "infer_encoder_config",
    "train_encoder",
]
