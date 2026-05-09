"""Beam-search validity tensors and helpers for SID-MLP inference."""

import torch

from .constants import CODEBOOK_SIZE, DIGIT_OFFSETS


@torch.inference_mode()
def build_valid_tensors(valid_set_path: str, device: torch.device):
    """
    Build dense masks for d2/d3, and sorted key tables for:
      - (d1,d2,d3)->valid d4 mask: d4_keys_sorted + d4_masks_sorted
      - (d1,d2,d3,d4)->item_idx:   keys4_sorted + vals4_sorted

    Returns 8-tuple:
      (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted,
       keys4_sorted, vals4_sorted, item_mapping, valid_items_4)
    """
    print("Loading valid_item_set.pt ...", flush=True)
    data = torch.load(valid_set_path, map_location="cpu", weights_only=False)

    valid_items_4 = data["valid_items_4"]
    item_mapping = data.get("item_mapping", None)

    if "d4_keys_sorted" not in data or "d4_masks_sorted" not in data:
        raise RuntimeError(
            "valid_item_set.pt missing d4_keys_sorted/d4_masks_sorted. "
            "Please regenerate valid_item_set.pt using the latest training script."
        )
    d4_keys_sorted_cpu = data["d4_keys_sorted"].long().contiguous()
    d4_masks_sorted_cpu = data["d4_masks_sorted"].bool().contiguous()

    print("Building dense masks d1->d2 and (d1,d2)->d3 ...", flush=True)
    mask_d1_d2 = torch.zeros((CODEBOOK_SIZE, CODEBOOK_SIZE), dtype=torch.bool)
    mask_d1d2_d3 = torch.zeros((CODEBOOK_SIZE, CODEBOOK_SIZE, CODEBOOK_SIZE), dtype=torch.bool)

    keys4 = []
    vals4 = []

    for (d1, d2, d3, d4), item_idx in valid_items_4.items():
        i1 = d1 - DIGIT_OFFSETS[0]
        i2 = d2 - DIGIT_OFFSETS[1]
        i3 = d3 - DIGIT_OFFSETS[2]
        i4 = d4 - DIGIT_OFFSETS[3]
        if not (0 <= i1 < 256 and 0 <= i2 < 256 and 0 <= i3 < 256 and 0 <= i4 < 256):
            continue

        mask_d1_d2[i1, i2] = True
        mask_d1d2_d3[i1, i2, i3] = True

        k4 = (((i1 * 256 + i2) * 256 + i3) * 256 + i4)
        keys4.append(k4)
        vals4.append(int(item_idx))

    keys4 = torch.tensor(keys4, dtype=torch.long)
    vals4 = torch.tensor(vals4, dtype=torch.long)

    sort_idx = torch.argsort(keys4)
    keys4_sorted = keys4[sort_idx].contiguous()
    vals4_sorted = vals4[sort_idx].contiguous()

    mask_d1_d2 = mask_d1_d2.to(device)
    mask_d1d2_d3 = mask_d1d2_d3.to(device)

    d4_keys_sorted = d4_keys_sorted_cpu.to(device)
    d4_masks_sorted = d4_masks_sorted_cpu.to(device)
    keys4_sorted = keys4_sorted.to(device)
    vals4_sorted = vals4_sorted.to(device)

    print(f"  mask_d1_d2 pairs: {int(mask_d1_d2.sum().item())}", flush=True)
    print(f"  mask_d1d2_d3 triplets: {int(mask_d1d2_d3.sum().item())}", flush=True)
    print(f"  d4 table size: {int(d4_keys_sorted.numel())}", flush=True)
    print(f"  item table size: {int(keys4_sorted.numel())}", flush=True)

    return (mask_d1_d2, mask_d1d2_d3, d4_keys_sorted, d4_masks_sorted,
            keys4_sorted, vals4_sorted, item_mapping, valid_items_4)


@torch.inference_mode()
def gather_d4_mask(keys3_sorted, masks3_sorted, key3_query):
    """Gather valid d4 masks for given (d1,d2,d3) triplet keys."""
    if keys3_sorted.numel() == 0:
        return torch.zeros((key3_query.numel(), CODEBOOK_SIZE), dtype=torch.bool, device=key3_query.device)
    idx = torch.searchsorted(keys3_sorted, key3_query).clamp(0, keys3_sorted.numel() - 1)
    found = keys3_sorted[idx] == key3_query
    out = torch.zeros((key3_query.numel(), CODEBOOK_SIZE), dtype=torch.bool, device=key3_query.device)
    if found.any():
        out[found] = masks3_sorted[idx[found]]
    return out


@torch.inference_mode()
def lookup_item_idx(keys4_sorted, vals4_sorted, key4_query):
    """Lookup item indices from quad keys."""
    if keys4_sorted.numel() == 0:
        return torch.full((key4_query.numel(),), -1, dtype=torch.long, device=key4_query.device)
    idx = torch.searchsorted(keys4_sorted, key4_query).clamp(0, keys4_sorted.numel() - 1)
    found = keys4_sorted[idx] == key4_query
    out = torch.full((key4_query.numel(),), -1, dtype=torch.long, device=key4_query.device)
    if found.any():
        out[found] = vals4_sorted[idx[found]]
    return out
