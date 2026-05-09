"""Build valid_item_set_{dataset_tag}.pt for SID-MLP beam validation/inference.

Usage:
  python -m sid_mlp.build_valid_set --graph_dir <path> --dataset_tag <tag>
"""

import argparse
import json
import os

from .constants import CODEBOOK_SIZE, DIGIT_OFFSETS, triplet_key_int, quad_key_int


def build(args):
    import torch

    if args.mapping_path:
        map_path = os.path.abspath(args.mapping_path)
    else:
        if not args.graph_dir:
            raise SystemExit('--graph_dir is required when --mapping_path is not set.')
        map_path = os.path.join(
            os.path.abspath(args.graph_dir),
            'hidden_states',
            args.dataset_tag,
            'item_mapping.json',
        )
    with open(map_path) as f:
        mapping = json.load(f)

    print(f"Building valid_item_set from {len(mapping)} items ...", flush=True)

    valid_items_4 = {}
    for item_idx_str, info in mapping.items():
        item_idx = int(item_idx_str)
        d1, d2, d3, d4 = [int(x) for x in info['semantic_id']]
        valid_items_4[(d1, d2, d3, d4)] = item_idx

    triplet_to_d4s = {}
    for (d1, d2, d3, d4), _item_idx in valid_items_4.items():
        i1 = d1 - DIGIT_OFFSETS[0]
        i2 = d2 - DIGIT_OFFSETS[1]
        i3 = d3 - DIGIT_OFFSETS[2]
        i4 = d4 - DIGIT_OFFSETS[3]

        if not (0 <= i1 < CODEBOOK_SIZE and 0 <= i2 < CODEBOOK_SIZE and
                0 <= i3 < CODEBOOK_SIZE and 0 <= i4 < CODEBOOK_SIZE):
            continue

        key3 = triplet_key_int(i1, i2, i3)
        if key3 not in triplet_to_d4s:
            triplet_to_d4s[key3] = torch.zeros(CODEBOOK_SIZE, dtype=torch.bool)
        triplet_to_d4s[key3][i4] = True

    sorted_triplet_keys = sorted(triplet_to_d4s.keys())
    d4_keys_sorted = torch.tensor(sorted_triplet_keys, dtype=torch.long)
    d4_masks_sorted = torch.stack([triplet_to_d4s[k] for k in sorted_triplet_keys])

    keys4, vals4 = [], []
    for (d1, d2, d3, d4), item_idx in valid_items_4.items():
        i1 = d1 - DIGIT_OFFSETS[0]
        i2 = d2 - DIGIT_OFFSETS[1]
        i3 = d3 - DIGIT_OFFSETS[2]
        i4 = d4 - DIGIT_OFFSETS[3]
        if not (0 <= i1 < CODEBOOK_SIZE and 0 <= i2 < CODEBOOK_SIZE and
                0 <= i3 < CODEBOOK_SIZE and 0 <= i4 < CODEBOOK_SIZE):
            continue
        keys4.append(quad_key_int(i1, i2, i3, i4))
        vals4.append(item_idx)

    sort_idx = torch.tensor(keys4).argsort()
    keys4_sorted = torch.tensor(keys4, dtype=torch.long)[sort_idx]
    vals4_sorted = torch.tensor(vals4, dtype=torch.long)[sort_idx]

    save_data = {
        'valid_items_4': valid_items_4,
        'item_mapping': {int(k): v for k, v in mapping.items()},
        'd4_keys_sorted': d4_keys_sorted,
        'd4_masks_sorted': d4_masks_sorted,
        'keys4_sorted': keys4_sorted,
        'vals4_sorted': vals4_sorted,
    }

    if args.output_path:
        out_path = os.path.abspath(args.output_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    else:
        if not args.graph_dir:
            raise SystemExit('--graph_dir is required when --output_path is not set.')
        out_dir = os.path.join(os.path.abspath(args.graph_dir), 'SID-MLP', 'ckpt')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'valid_item_set_{args.dataset_tag}.pt')
    torch.save(save_data, out_path)

    print(f"Saved: {out_path}", flush=True)
    print(f"  items={len(valid_items_4)} triplets={len(triplet_to_d4s)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_tag', type=str, required=True)
    parser.add_argument('--graph_dir', type=str, default=os.environ.get('SID_MLP_GRAPH_DIR'))
    parser.add_argument('--mapping_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
