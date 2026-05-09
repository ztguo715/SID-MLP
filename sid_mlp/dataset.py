"""GPU-resident data packing for the public SID-MLP path."""

import gc
import json
import os
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .constants import DIGIT_OFFSETS, token_to_codebook


class CommonLensDataset(Dataset):
    """GPU-resident dataset used by SID-MLP training."""

    def __init__(self, enc_seq, enc_mask, hk, digits, teacher_logits, gt_digits):
        self.enc_seq = enc_seq
        self.enc_mask = enc_mask
        self.hk = hk
        self.digits = digits
        self.teacher_logits = teacher_logits
        self.gt_digits = gt_digits

    def __len__(self):
        return self.enc_seq.shape[0]

    def __getitem__(self, idx):
        return idx


def collate_fn_gpu(dataset: CommonLensDataset, indices: List[int]):
    idx = torch.tensor(indices, dtype=torch.long, device=dataset.enc_seq.device)

    enc_seq = dataset.enc_seq[idx].float()
    enc_mask = dataset.enc_mask[idx]
    hk = None

    digits = dataset.digits[idx]
    d1, d2, d3, d4 = digits[:, 0], digits[:, 1], digits[:, 2], digits[:, 3]

    tl = dataset.teacher_logits[idx].float()
    tl1, tl2, tl3, tl4 = tl[:, 0], tl[:, 1], tl[:, 2], tl[:, 3]

    gt = dataset.gt_digits[idx]
    gd1, gd2, gd3, gd4 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]

    return enc_seq, enc_mask, hk, d1, d2, d3, d4, tl1, tl2, tl3, tl4, gd1, gd2, gd3, gd4


def _pack_from_raw(graph_dir: str, dataset_tag: str, storage_dtype: torch.dtype) -> dict:
    enc_path = os.path.join(graph_dir, "hidden_states", dataset_tag, "encoder_sequences.pt")
    log_path = os.path.join(graph_dir, "logits", dataset_tag, "logits.pt")
    map_path = os.path.join(graph_dir, "hidden_states", dataset_tag, "item_mapping.json")

    enc_dict = torch.load(enc_path, map_location="cpu", weights_only=False, mmap=True)
    logits_dict = torch.load(log_path, map_location="cpu", weights_only=False, mmap=True)

    with open(map_path) as f:
        mapping = json.load(f)
    item_sem = {int(k): v["semantic_id"] for k, v in mapping.items()}

    total_samples = 0
    max_seq_len = 0
    skipped = 0
    valid_items = []

    for idx in tqdm(list(enc_dict.keys()), desc="Counting samples", ncols=100):
        if idx not in logits_dict or idx not in item_sem:
            skipped += 1
            continue

        gt_d1, gt_d2, gt_d3, gt_d4 = item_sem[idx]
        gd1 = token_to_codebook(gt_d1, 0)
        gd2 = token_to_codebook(gt_d2, 1)
        gd3 = token_to_codebook(gt_d3, 2)
        gd4 = token_to_codebook(gt_d4, 3)

        n_ctx = min(len(enc_dict[idx]), logits_dict[idx].shape[0])
        for c in range(n_ctx):
            enc_c = enc_dict[idx][c] if isinstance(enc_dict[idx], list) else enc_dict[idx][c]
            max_seq_len = max(max_seq_len, enc_c.shape[0])

        total_samples += n_ctx
        valid_items.append((idx, n_ctx, gt_d1, gt_d2, gt_d3, gt_d4, gd1, gd2, gd3, gd4))

    print(f"Found {total_samples} samples, max_seq_len={max_seq_len}, skipped items={skipped}", flush=True)

    enc_seq_all = torch.zeros(total_samples, max_seq_len, 128, dtype=storage_dtype)
    enc_lens = torch.zeros(total_samples, dtype=torch.int32)
    digits_all = torch.zeros(total_samples, 4, dtype=torch.int64)
    tl_all = torch.zeros(total_samples, 4, 256, dtype=storage_dtype)
    gt_all = torch.zeros(total_samples, 4, dtype=torch.int64)

    sample_idx = 0
    for idx, n_ctx, gt_d1, gt_d2, gt_d3, gt_d4, gd1, gd2, gd3, gd4 in tqdm(
        valid_items, desc="Packing tensors", ncols=100
    ):
        for c in range(n_ctx):
            enc_c = enc_dict[idx][c]
            sl = enc_c.shape[0]
            enc_seq_all[sample_idx, :sl, :] = (
                enc_c.to(storage_dtype) if enc_c.dim() == 2 else enc_c.unsqueeze(0).to(storage_dtype)
            )
            enc_lens[sample_idx] = sl
            digits_all[sample_idx] = torch.tensor([gt_d1, gt_d2, gt_d3, gt_d4], dtype=torch.int64)

            full = logits_dict[idx][c]
            for d in range(4):
                start = DIGIT_OFFSETS[d]
                tl_all[sample_idx, d, :] = full[d, start:start + 256].to(storage_dtype)

            gt_all[sample_idx] = torch.tensor([gd1, gd2, gd3, gd4], dtype=torch.int64)
            sample_idx += 1

    del enc_dict, logits_dict
    gc.collect()

    arange = torch.arange(max_seq_len).unsqueeze(0)
    enc_mask_all = (arange >= enc_lens.unsqueeze(1)).to(torch.bool)
    del enc_lens

    return {
        "enc_seq": enc_seq_all,
        "enc_mask": enc_mask_all,
        "digits": digits_all,
        "teacher_logits": tl_all,
        "gt_digits": gt_all,
    }


def _cache_path(graph_dir: str, dataset_tag: str, storage_dtype: torch.dtype) -> str:
    dtype_tag = "fp16" if storage_dtype == torch.float16 else "fp32"
    cache_dir = os.path.join(graph_dir, "hidden_states", dataset_tag, "packed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"k0_{dtype_tag}.pt")


def prepare_common_data(graph_dir: str, dataset_tag: str, device: torch.device,
                        storage_dtype: torch.dtype = torch.float32):
    cache_file = _cache_path(graph_dir, dataset_tag, storage_dtype)

    if os.path.exists(cache_file):
        print(f"Loading packed cache (mmap): {cache_file}", flush=True)
        cached = torch.load(cache_file, map_location="cpu", weights_only=False, mmap=True)
    else:
        print("No cache found, packing from raw data ...", flush=True)
        cached = _pack_from_raw(graph_dir, dataset_tag, storage_dtype)
        print(f"Saving packed cache: {cache_file}", flush=True)
        torch.save(cached, cache_file)
        del cached
        gc.collect()
        cached = torch.load(cache_file, map_location="cpu", weights_only=False, mmap=True)

    total_samples = cached["enc_seq"].shape[0]
    print(f"Moving {total_samples} samples to {device} ...", flush=True)
    enc_seq_gpu = cached["enc_seq"].to(device, non_blocking=True)
    enc_mask_gpu = cached["enc_mask"].to(device, non_blocking=True)
    digits_gpu = cached["digits"].to(device, non_blocking=True)
    tl_gpu = cached["teacher_logits"].to(device, non_blocking=True)
    gt_gpu = cached["gt_digits"].to(device, non_blocking=True)

    del cached
    gc.collect()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        allocated_gb = torch.cuda.memory_allocated(device) / 1e9
        print(f"GPU memory after data load: {allocated_gb:.2f} GB", flush=True)

    return CommonLensDataset(enc_seq_gpu, enc_mask_gpu, None, digits_gpu, tl_gpu, gt_gpu)
