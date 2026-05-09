"""Training loop for the public SID-MLP decoder."""

import datetime as dt
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .beam import build_valid_tensors
from .constants import set_seed
from .dataset import collate_fn_gpu, prepare_common_data
from .inference import (
    build_tiger_pipeline,
    evaluate_loader,
    find_valid_set,
    replace_t5_encoder_with_sidmlp_pp,
)
from .losses import d4_loss_fn, loss_fn, topk_correct
from .models import SIDMLP

BEAM_VAL_INTERVAL = 5


def _fmt_float(value):
    text = f"{value:e}"
    mantissa, exponent = text.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exponent)
    return f"{mantissa}e{exp}" if exp >= 0 else f"{mantissa}e-{abs(exp)}"


def _save_prefix(args):
    attn = f"_ad{args.attn_dim}" if args.attn_dim > 0 else ""
    variant = "sidmlpp" if getattr(args, "encoder_ckpt", None) else "sidmlp"
    base = (
        f"{args.category}_{variant}_k0_xattn"
        f"_h{args.num_heads}_f{args.ffn_dim}{attn}"
        f"_hhid{args.head_hidden}_hl{args.head_layers}"
        f"_lr{_fmt_float(args.lr)}"
    )
    if args.run_tag:
        base += f"_{args.run_tag}"
    return base


def _checkpoint_args(args):
    return {
        key: value
        for key, value in vars(args).items()
        if not key.startswith("_") and not callable(value)
    }


def _batch_loss(args, model, batch):
    enc_seq, enc_mask, _hk, d1, d2, d3, _d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4 = batch
    logits1, logits2, logits3, logits4 = model(enc_seq, enc_mask, d1_ids=d1, d2_ids=d2, d3_ids=d3)
    loss = (
        loss_fn(args, logits1, tl1, gd1)
        + loss_fn(args, logits2, tl2, gd2)
        + loss_fn(args, logits3, tl3, gd3)
        + d4_loss_fn(args, logits4, tl4, gd4)
    )
    return loss, (logits1, logits2, logits3, logits4), (gd1, gd2, gd3, gd4)


@torch.no_grad()
def _validate_digits(args, model, loader, device):
    model.eval()
    losses = []
    correct_at1 = np.zeros(4, dtype=np.int64)
    correct_at5 = np.zeros(4, dtype=np.int64)
    total = 0

    for batch in tqdm(loader, desc="Val digits", ncols=100):
        loss, logits, labels = _batch_loss(args, model, batch)
        losses.append(float(loss.item()))
        total += labels[-1].shape[0]
        for idx, (logit, label) in enumerate(zip(logits, labels)):
            c1, c5 = topk_correct(logit, label, k=5)
            correct_at1[idx] += c1
            correct_at5[idx] += c5

    return {
        "loss": float(np.mean(losses)) if losses else float("inf"),
        "acc1": correct_at1 / max(total, 1),
        "acc5": correct_at5 / max(total, 1),
    }


def _setup_beam_validation(args, device):
    pipeline = build_tiger_pipeline(args)
    t5 = pipeline.model.t5.to(device).eval()
    for param in t5.parameters():
        param.requires_grad_(False)

    if args.encoder_ckpt:
        replace_t5_encoder_with_sidmlp_pp(
            t5,
            args.encoder_ckpt,
            device,
        )
        if not getattr(t5.encoder, "sidmlp_pp_patched", False):
            raise RuntimeError("SID-MLP++ validation failed to patch the TIGER teacher encoder.")
        print(f"Patched TIGER teacher encoder with SID-MLP++ encoder for validation: {args.encoder_ckpt}", flush=True)

    split = "val" if "val" in pipeline.tokenized_datasets else "test"
    loader = DataLoader(
        pipeline.tokenized_datasets[split],
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[split],
        num_workers=0,
        pin_memory=False,
    )
    valid_set = find_valid_set(
        args.dataset_tag,
        explicit_path=args.valid_set,
        graph_dir=args.graph_dir,
    )
    if valid_set is None:
        raise FileNotFoundError(
            f"Cannot find valid_item_set for dataset_tag={args.dataset_tag}. "
            "Run build_valid_set.py or pass --valid_set."
        )
    return t5, loader, build_valid_tensors(valid_set, device)


def _check_sidmlp_pp_training_tags(args):
    if not getattr(args, "encoder_ckpt", None):
        return
    base_tag = f"{args.dataset}_{args.category}"
    if args.dataset_tag == base_tag or args.val_dataset_tag == f"{base_tag}_val":
        raise SystemExit(
            "SID-MLP++ stage2 must train on transformed encoder features, not the base TIGER "
            "hidden-state tags. Run scripts/extract.sh sidmlp-pp first and use the transformed dataset_tag."
        )


def train_sidmlp(args):
    _check_sidmlp_pp_training_tags(args)
    if not getattr(args, "graph_dir", None):
        raise SystemExit("--graph_dir is required. Set it in the launcher script or pass it on the CLI.")
    graph_dir = os.path.abspath(args.graph_dir)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.abspath(args.ckpt_dir or os.path.join(graph_dir, "SID-MLP", "ckpt"))
    log_dir = os.path.abspath(args.log_dir or os.path.join(graph_dir, "SID-MLP", "logs"))
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    save_prefix = _save_prefix(args)
    log_path = os.path.join(log_dir, f"{save_prefix}_{dt.datetime.now():%Y%m%d_%H%M%S}.log")

    def log(message):
        print(message, flush=True)
        with open(log_path, "a") as handle:
            print(message, file=handle)

    teacher_ckpt = torch.load(os.path.abspath(args.checkpoint), map_location="cpu", weights_only=False)
    embedding_weight = teacher_ckpt["t5.shared.weight"]
    del teacher_ckpt

    storage_dtype = torch.float16 if args.storage_fp16 else torch.float32
    train_ds = prepare_common_data(graph_dir, args.dataset_tag, device, storage_dtype)
    val_ds = prepare_common_data(graph_dir, args.val_dataset_tag, device, storage_dtype)
    make_collate = lambda dataset: (lambda indices: collate_fn_gpu(dataset, indices))
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=make_collate(train_ds),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=make_collate(val_ds),
    )

    model = SIDMLP(
        embedding_weight,
        embed_dim=embedding_weight.shape[1],
        d_model=128,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        attn_dim=args.attn_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    beam_t5 = beam_loader = beam_valid_tensors = None
    if args.epochs >= BEAM_VAL_INTERVAL:
        beam_t5, beam_loader, beam_valid_tensors = _setup_beam_validation(args, device)

    log(f"Training {'SID-MLP++' if args.encoder_ckpt else 'SID-MLP'}")
    log(f"train={len(train_ds)} val={len(val_ds)} checkpoint_dir={ckpt_dir}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train {epoch}/{args.epochs}", ncols=100)):
            if args.test and batch_idx >= 5:
                break
            optimizer.zero_grad(set_to_none=True)
            loss, _logits, _labels = _batch_loss(args, model, batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))

        scheduler.step()
        digit_metrics = _validate_digits(args, model, val_loader, device)
        beam_metrics = {}
        if beam_t5 is not None and epoch % BEAM_VAL_INTERVAL == 0:
            beam_metrics = evaluate_loader(
                model,
                beam_t5,
                beam_loader,
                beam_valid_tensors,
                device,
                beam_size=args.beam_size,
                final_top_n=args.final_top_n,
                max_batches=5 if args.test else None,
            )

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        log(
            f"epoch={epoch} train_loss={train_loss:.6f} val_loss={digit_metrics['loss']:.6f} "
            f"d1@1={digit_metrics['acc1'][0]:.4f} d2@1={digit_metrics['acc1'][1]:.4f} "
            f"d3@1={digit_metrics['acc1'][2]:.4f} d4@1={digit_metrics['acc1'][3]:.4f} "
            f"ndcg10={beam_metrics.get('ndcg10', 0.0):.6f}"
        )

    final_path = os.path.join(ckpt_dir, f"{save_prefix}_final.pth")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "args": _checkpoint_args(args),
        "model_type": "SID-MLP++" if args.encoder_ckpt else "SID-MLP",
    }, final_path)
    log(f"Saved final checkpoint: {final_path}")
