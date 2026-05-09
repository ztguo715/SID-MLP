"""SID-MLP and SID-MLP++ beam-search inference."""

import glob
import math
import os
import sys
import time
from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .beam import build_valid_tensors, gather_d4_mask, lookup_item_idx
from .constants import CODEBOOK_SIZE, DIGIT_OFFSETS, quad_key_int, triplet_key_int
from .models import SIDMLP
from .pretrain_encoder import build_encoder, infer_encoder_config

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKOUT_ROOT = os.path.abspath(os.path.join(PACKAGE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.environ.get("PROJECT_DIR", os.path.join(CHECKOUT_ROOT, "..")))


def _ensure_import_paths():
    for path in (PROJECT_ROOT, CHECKOUT_ROOT):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def _torch_load(path, *, weights_only):
    try:
        return torch.load(os.path.abspath(path), map_location="cpu", weights_only=weights_only)
    except TypeError:
        return torch.load(os.path.abspath(path), map_location="cpu")


def build_tiger_pipeline(args):
    _ensure_import_paths()
    from genrec.pipeline import Pipeline

    if not getattr(args, "sem_ids", None):
        raise ValueError("--sem_ids is required. Set it in the launcher script or pass it on the CLI.")
    sem_ids_path = os.path.abspath(args.sem_ids)
    return Pipeline(
        model_name="TIGER",
        dataset_name=args.dataset,
        checkpoint_path=os.path.abspath(args.checkpoint),
        config_dict={
            "category": args.category,
            "custom_sem_ids_path": sem_ids_path,
            "run_id": "sid_mlp_public_infer",
        },
    )


def build_sidmlp_from_checkpoint(checkpoint_path, embedding_weight, device, *, loaded_checkpoint=None):
    ckpt = loaded_checkpoint if loaded_checkpoint is not None else _torch_load(checkpoint_path, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        ckpt_args = ckpt.get("args", {})
    else:
        state = ckpt
        ckpt_args = {}

    inferred_attn_dim = int(ckpt_args.get("attn_dim", state["cross_attn.q_proj.weight"].shape[0]))
    inferred_ffn_dim = int(ckpt_args.get("ffn_dim", state["ffn.0.weight"].shape[0]))
    inferred_head_hidden = int(ckpt_args.get("head_hidden", state["head1.0.weight"].shape[0]))

    model = SIDMLP(
        embedding_weight,
        embed_dim=embedding_weight.shape[1],
        d_model=128,
        num_heads=int(ckpt_args.get("num_heads", 4)),
        ffn_dim=inferred_ffn_dim,
        dropout=float(ckpt_args.get("dropout", 0.0)),
        head_hidden=inferred_head_hidden,
        head_layers=int(ckpt_args.get("head_layers", 1)),
        attn_dim=inferred_attn_dim,
    )
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, ckpt


def configure_precision(args, device):
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.set_float32_matmul_precision("highest")
        except Exception:
            pass

    native_bf16 = device.type == "cuda" and not getattr(args, "fp32", False)
    if native_bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Default native bf16 inference requires CUDA bf16 support. Pass --fp32 to run fp32.")
    return native_bf16


def load_sidmlp_pp_encoder(
    path,
    device,
    *,
    native_bf16=False,
):
    raw_state = _torch_load(path, weights_only=True)
    ckpt_args = {}
    if isinstance(raw_state, dict) and "model_state_dict" in raw_state:
        ckpt_args = raw_state.get("args", {})
        state = raw_state["model_state_dict"]
    else:
        state = raw_state
    encoder = build_encoder(**infer_encoder_config(state, ckpt_args))
    encoder.load_state_dict(state, strict=True)
    encoder.to(device).eval()
    if native_bf16:
        encoder = encoder.bfloat16()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


class _SIDMLPPlusPlusEncoderWrapper(torch.nn.Module):
    def __init__(self, shared, encoder):
        super().__init__()
        self.shared = shared
        self.encoder = encoder
        self.sidmlp_pp_patched = True

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        raw_embeddings = self.shared(input_ids)
        encoder_dtype = next(self.encoder.parameters()).dtype
        if raw_embeddings.dtype != encoder_dtype:
            raw_embeddings = raw_embeddings.to(encoder_dtype)
        pad_mask = (~attention_mask.bool()) if attention_mask is not None else None
        hidden = self.encoder(raw_embeddings, src_key_padding_mask=pad_mask)
        return SimpleNamespace(last_hidden_state=hidden)


def replace_t5_encoder_with_sidmlp_pp(
    t5,
    encoder_ckpt,
    device,
    *,
    native_bf16=False,
):
    encoder = load_sidmlp_pp_encoder(
        encoder_ckpt,
        device,
        native_bf16=native_bf16,
    )
    t5.encoder = _SIDMLPPlusPlusEncoderWrapper(t5.shared, encoder).to(device).eval()
    return encoder


def _is_sidmlp_pp_checkpoint(ckpt, lens_path):
    if isinstance(ckpt, dict):
        model_type = str(ckpt.get("model_type", "")).lower()
        if "sid-mlp++" in model_type or "sidmlp++" in model_type or "sidmlpp" in model_type:
            return True
        ckpt_args = ckpt.get("args", {})
        if isinstance(ckpt_args, dict) and ckpt_args.get("encoder_ckpt"):
            return True
    filename = os.path.basename(str(lens_path)).lower()
    return "sidmlpp" in filename or "sidmlp_pp" in filename


def _resolve_sidmlp_pp_encoder(args, lens_ckpt):
    if getattr(args, "encoder_ckpt", None):
        return os.path.abspath(args.encoder_ckpt)

    if isinstance(lens_ckpt, dict):
        ckpt_args = lens_ckpt.get("args", {})
        if isinstance(ckpt_args, dict):
            saved_path = ckpt_args.get("encoder_ckpt")
            if saved_path:
                return os.path.abspath(saved_path)
    return None


def _require_sidmlp_pp_patch(args, lens_ckpt):
    encoder_ckpt = _resolve_sidmlp_pp_encoder(args, lens_ckpt)
    sidmlp_pp = bool(encoder_ckpt) or _is_sidmlp_pp_checkpoint(lens_ckpt, args.lens_path)
    if sidmlp_pp and not encoder_ckpt:
        raise RuntimeError(
            "SID-MLP++ inference requires --encoder_ckpt. Refusing to run with the TIGER teacher encoder."
        )
    if encoder_ckpt and not os.path.exists(encoder_ckpt):
        raise FileNotFoundError(
            f"SID-MLP++ encoder checkpoint not found: {encoder_ckpt}. "
            "Pass --encoder_ckpt explicitly to avoid falling back to the teacher encoder."
        )
    return encoder_ckpt


def find_valid_set(dataset_tag, *, explicit_path=None, lens_path=None, graph_dir=None):
    filename = f"valid_item_set_{dataset_tag}.pt"
    candidates = []
    if explicit_path:
        candidates.append(os.path.abspath(explicit_path))
    if lens_path:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(lens_path)), filename))
    if graph_dir:
        graph_dir = os.path.abspath(graph_dir)
        candidates.extend([
            os.path.join(graph_dir, "SID-MLP", "ckpt", filename),
            os.path.join(graph_dir, "SID-MLP", filename),
            os.path.join(graph_dir, filename),
        ])
    for path in candidates:
        if path and os.path.exists(path):
            return path

    if graph_dir:
        pattern = os.path.join(graph_dir, "SID-MLP", "ckpt", f"valid_item_set_{dataset_tag}*.pt")
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def _dedup_rank(row_ids, gt_item_idx, limit):
    seen = set()
    dedup = []
    for item_idx in row_ids[:limit]:
        if item_idx == -1 or item_idx in seen:
            continue
        seen.add(item_idx)
        dedup.append(item_idx)
    try:
        return dedup.index(gt_item_idx)
    except ValueError:
        return -1


def _resolve_ground_truth_items(labels, valid_items_4):
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu()

    gt_sids = []
    gt_item_indices = []
    missing = []
    for row in range(len(labels)):
        values = []
        for pos in range(4):
            value = labels[row][pos]
            values.append(int(value.item() if hasattr(value, "item") else value))
        gt = tuple(values)
        gt_item_idx = valid_items_4.get(gt)
        if gt_item_idx is None:
            missing.append(gt)
        else:
            gt_sids.append(gt)
            gt_item_indices.append(int(gt_item_idx))

    if missing:
        raise RuntimeError(
            "Ground-truth SID tokens are missing from the catalog validity mask. "
            f"missing={len(missing)} examples={missing[:5]}. "
            "Check that --valid_set, --dataset_tag, --sem_ids, and category all match."
        )
    return gt_sids, gt_item_indices


@torch.inference_mode()
def evaluate_loader(
    lens,
    t5,
    loader,
    valid_tensors,
    device,
    *,
    beam_size=50,
    final_top_n=50,
    d1_temp=1.0,
    d2_temp=1.0,
    d3_temp=1.0,
    d4_temp=1.0,
    alpha_d2=1.0,
    gamma_d3=1.0,
    delta_d4=1.0,
    max_batches=None,
):
    (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted,
     keys4_sorted, vals4_sorted, _item_mapping, valid_items_4) = valid_tensors

    recall = defaultdict(int)
    ndcg = defaultdict(float)
    coverage = defaultdict(int)
    total = 0
    test_ns = (1, 5, 10)
    beam = int(beam_size)
    final_n = int(final_top_n)
    d1_temp = max(float(d1_temp), 1e-6)
    d2_temp = max(float(d2_temp), 1e-6)
    d3_temp = max(float(d3_temp), 1e-6)
    d4_temp = max(float(d4_temp), 1e-6)
    alpha_d2 = float(alpha_d2)
    gamma_d3 = float(gamma_d3)
    delta_d4 = float(delta_d4)

    lens.eval()
    t5.eval()

    for batch_idx, batch in enumerate(tqdm(loader, desc="SID-MLP beam", ncols=100)):
        if max_batches is not None and batch_idx >= max_batches:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        gt_sids, gt_item_indices = _resolve_ground_truth_items(batch["labels"], valid_items_4)
        batch_size = input_ids.shape[0]

        enc_out = t5.encoder(input_ids=input_ids, attention_mask=attn_mask, return_dict=True)
        enc_hidden = enc_out.last_hidden_state
        enc_pad_mask = ~attn_mask.bool()

        ctx = lens.encode(enc_hidden, enc_pad_mask)
        logits1 = lens.head1(ctx).float()
        d1_lp = F.log_softmax(logits1 / d1_temp, dim=-1)
        d1_scores, d1_idx = torch.topk(d1_lp, beam, dim=1)
        d1_tokens = d1_idx + DIGIT_OFFSETS[0]

        ctx_view = ctx.unsqueeze(1).expand(-1, beam, -1).contiguous()
        d1_flat = d1_tokens.reshape(batch_size * beam)
        ctx_flat = ctx_view.reshape(batch_size * beam, -1)
        logits2 = lens.head2(torch.cat([ctx_flat, lens.embedding(d1_flat)], dim=1)).float()
        d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
        logits2 = logits2.masked_fill(~mask_d1_d2[d1_cb], float("-inf"))
        d2_lp = F.log_softmax(logits2 / d2_temp, dim=-1).view(
            batch_size, beam, CODEBOOK_SIZE
        )

        score2 = d1_scores.unsqueeze(-1) + alpha_d2 * d2_lp
        best2_score, best2_idx = torch.topk(
            score2.masked_fill(torch.isnan(score2), float("-inf")).view(batch_size, -1),
            beam,
            dim=1,
        )
        sel_beam2 = best2_idx // CODEBOOK_SIZE
        sel_d2 = best2_idx % CODEBOOK_SIZE
        d1_tokens = torch.gather(d1_tokens, 1, sel_beam2)
        d2_tokens = sel_d2 + DIGIT_OFFSETS[1]
        prefix_score = best2_score

        d1_flat = d1_tokens.reshape(batch_size * beam)
        d2_flat = d2_tokens.reshape(batch_size * beam)
        ctx_flat = ctx_view.reshape(batch_size * beam, -1)
        logits3 = lens.head3(torch.cat([
            ctx_flat,
            lens.embedding(d1_flat),
            lens.embedding(d2_flat),
        ], dim=1)).float()
        d1_cb = (d1_flat - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1)
        d2_cb = (d2_flat - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE - 1)
        logits3 = logits3.masked_fill(~mask_d1d2_d3[d1_cb, d2_cb], float("-inf"))
        d3_lp = F.log_softmax(logits3 / d3_temp, dim=-1).view(
            batch_size, beam, CODEBOOK_SIZE
        )

        score3 = prefix_score.unsqueeze(-1) + gamma_d3 * d3_lp
        best3_score, best3_idx = torch.topk(
            score3.masked_fill(torch.isnan(score3), float("-inf")).view(batch_size, -1),
            beam,
            dim=1,
        )
        sel_beam3 = best3_idx // CODEBOOK_SIZE
        sel_d3 = best3_idx % CODEBOOK_SIZE
        d1_tokens = torch.gather(d1_tokens, 1, sel_beam3)
        d2_tokens = torch.gather(d2_tokens, 1, sel_beam3)
        d3_tokens = sel_d3 + DIGIT_OFFSETS[2]

        d1_flat = d1_tokens.reshape(batch_size * beam)
        d2_flat = d2_tokens.reshape(batch_size * beam)
        d3_flat = d3_tokens.reshape(batch_size * beam)
        ctx_flat = ctx_view.reshape(batch_size * beam, -1)
        logits4 = lens.head4(torch.cat([
            ctx_flat,
            lens.embedding(d1_flat),
            lens.embedding(d2_flat),
            lens.embedding(d3_flat),
        ], dim=1)).float()

        key3 = triplet_key_int(
            d1_flat - DIGIT_OFFSETS[0],
            d2_flat - DIGIT_OFFSETS[1],
            d3_flat - DIGIT_OFFSETS[2],
        )
        logits4 = logits4.masked_fill(~gather_d4_mask(d4_keys_sorted, d4_masks_sorted, key3), float("-inf"))
        d4_lp = F.log_softmax(logits4 / d4_temp, dim=-1)
        top_d4_val, top_d4_idx = torch.topk(d4_lp, CODEBOOK_SIZE, dim=1)
        d4_tokens = top_d4_idx + DIGIT_OFFSETS[3]

        quad_scores = best3_score.unsqueeze(-1) + delta_d4 * top_d4_val.view(
            batch_size, beam, CODEBOOK_SIZE
        )
        topq_idx = torch.topk(
            quad_scores.masked_fill(torch.isnan(quad_scores), float("-inf")).view(batch_size, -1),
            final_n,
            dim=1,
        ).indices
        beam_idx = topq_idx // CODEBOOK_SIZE

        d1_final = torch.gather(d1_tokens, 1, beam_idx)
        d2_final = torch.gather(d2_tokens, 1, beam_idx)
        d3_final = torch.gather(d3_tokens, 1, beam_idx)
        d4_final = torch.gather(d4_tokens.view(batch_size, -1), 1, topq_idx)

        key4 = quad_key_int(
            (d1_final - DIGIT_OFFSETS[0]).clamp(0, CODEBOOK_SIZE - 1),
            (d2_final - DIGIT_OFFSETS[1]).clamp(0, CODEBOOK_SIZE - 1),
            (d3_final - DIGIT_OFFSETS[2]).clamp(0, CODEBOOK_SIZE - 1),
            (d4_final - DIGIT_OFFSETS[3]).clamp(0, CODEBOOK_SIZE - 1),
        ).view(batch_size * final_n)
        item_ids = lookup_item_idx(keys4_sorted, vals4_sorted, key4).view(batch_size, final_n)

        item_rows = item_ids.detach().cpu().tolist()
        d1_beams = d1_tokens.detach().cpu().tolist()
        d2_beams = d2_tokens.detach().cpu().tolist()
        d3_beams = d3_tokens.detach().cpu().tolist()

        for row in range(batch_size):
            gt = gt_sids[row]
            gt_item_idx = gt_item_indices[row]
            total += 1
            rank = _dedup_rank(item_rows[row], gt_item_idx, final_n)
            for cutoff in test_ns:
                if rank != -1 and rank < cutoff:
                    recall[cutoff] += 1
                    ndcg[cutoff] += 1.0 / math.log2(rank + 2)

            if gt[0] in d1_beams[row]:
                coverage["d1"] += 1
            if any(d1_beams[row][i] == gt[0] and d2_beams[row][i] == gt[1] for i in range(beam)):
                coverage["d1d2"] += 1
            if any(
                d1_beams[row][i] == gt[0]
                and d2_beams[row][i] == gt[1]
                and d3_beams[row][i] == gt[2]
                for i in range(beam)
            ):
                coverage["d1d2d3"] += 1
            if rank != -1:
                coverage["item"] += 1

    metrics = {"total": total}
    for cutoff in test_ns:
        metrics[f"recall{cutoff}"] = recall[cutoff] / max(total, 1)
        metrics[f"ndcg{cutoff}"] = ndcg[cutoff] / max(total, 1)
    for key, value in coverage.items():
        metrics[f"{key}_hit"] = value / max(total, 1)
    return metrics


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    native_bf16 = configure_precision(args, device)
    lens_ckpt = _torch_load(args.lens_path, weights_only=False)
    encoder_ckpt = _require_sidmlp_pp_patch(args, lens_ckpt)
    args.encoder_ckpt = encoder_ckpt

    pipeline = build_tiger_pipeline(args)
    tiger_model = pipeline.model.to(device).eval()
    t5 = tiger_model.t5
    if native_bf16:
        t5.bfloat16()
    for param in t5.parameters():
        param.requires_grad_(False)

    if encoder_ckpt:
        replace_t5_encoder_with_sidmlp_pp(
            t5,
            encoder_ckpt,
            device,
            native_bf16=native_bf16,
        )
        if not getattr(t5.encoder, "sidmlp_pp_patched", False):
            raise RuntimeError("SID-MLP++ inference failed to patch the TIGER teacher encoder.")
        print(f"Patched TIGER teacher encoder with SID-MLP++ encoder: {encoder_ckpt}", flush=True)

    teacher_ckpt = _torch_load(args.checkpoint, weights_only=False)
    lens, lens_ckpt = build_sidmlp_from_checkpoint(
        args.lens_path,
        teacher_ckpt["t5.shared.weight"],
        device,
        loaded_checkpoint=lens_ckpt,
    )
    if native_bf16:
        lens.bfloat16()
    del teacher_ckpt

    valid_set = find_valid_set(
        args.dataset_tag,
        explicit_path=args.valid_set,
        lens_path=args.lens_path,
        graph_dir=getattr(args, "graph_dir", None),
    )
    if valid_set is None:
        raise FileNotFoundError(f"Cannot find valid_item_set for dataset_tag={args.dataset_tag}")
    valid_tensors = build_valid_tensors(valid_set, device)

    split = args.split or ("val" if "val" in pipeline.tokenized_datasets else "test")
    dataset = pipeline.tokenized_datasets[split]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pipeline.tokenizer.collate_fn[split],
        num_workers=0,
        pin_memory=False,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.time()
    metrics = evaluate_loader(
        lens,
        t5,
        loader,
        valid_tensors,
        device,
        beam_size=args.beam_size,
        final_top_n=args.final_top_n,
        d1_temp=args.d1_temp,
        d2_temp=args.d2_temp,
        d3_temp=args.d3_temp,
        d4_temp=args.d4_temp,
        alpha_d2=args.alpha_d2,
        gamma_d3=args.gamma_d3,
        delta_d4=args.delta_d4,
        max_batches=5 if args.test else None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        metrics["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    metrics["elapsed_sec"] = time.time() - start
    metrics["throughput"] = metrics["total"] / max(metrics["elapsed_sec"], 1e-6)

    print(
        f"SID-MLP{'++' if args.encoder_ckpt else ''}: "
        f"split={split} total={metrics['total']} "
        f"R@10={metrics['recall10']:.4f} N@10={metrics['ndcg10']:.4f} "
        f"throughput={metrics['throughput']:.1f}/s "
        f"dtype={'native-bfloat16' if native_bf16 else 'fp32'} tf32=False",
        flush=True,
    )
    if "peak_mem_gb" in metrics:
        print(f"peak_mem={metrics['peak_mem_gb']:.3f} GB", flush=True)
    return metrics, lens_ckpt
