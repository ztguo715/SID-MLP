"""Extract TIGER features for SID-MLP training."""

import argparse
import json
import os
import sys
from collections import defaultdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CHECKOUT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_PROJECT_ROOT = os.path.abspath(os.environ.get('PROJECT_DIR', os.path.join(_CHECKOUT_ROOT, '..')))


def _ensure_import_paths():
    for path in (_PROJECT_ROOT, _CHECKOUT_ROOT):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def build_reverse_mapping(tokenizer, dataset):
    """Build token_tuple → 0-based item_idx and item_idx → info mappings."""
    item2id = dataset.item2id
    tokens_to_item_idx = {}
    item_idx_to_info = {}

    for item_str, token_tuple in tokenizer.item2tokens.items():
        if item_str not in item2id:
            continue
        zero_idx = item2id[item_str] - 1
        tokens_to_item_idx[token_tuple] = zero_idx
        item_idx_to_info[zero_idx] = {
            'original_id': item_str,
            'semantic_id': list(token_tuple),
        }

    return tokens_to_item_idx, item_idx_to_info


def _save_item_mapping(item_idx_to_info, output_dir):
    mapping_path = os.path.join(output_dir, 'item_mapping.json')
    mapping_json = {str(k): v for k, v in item_idx_to_info.items()}
    with open(mapping_path, 'w') as f:
        json.dump(mapping_json, f, indent=2)
    print(f"  Saved item mapping: {mapping_path} ({len(mapping_json)} items)")


def _print_context_stats(data_dict, name):
    import numpy as np
    import torch

    counts = [v.shape[0] if isinstance(v, torch.Tensor) else len(v)
              for v in data_dict.values()]
    print(f"  {name}: {len(data_dict)} items, "
          f"contexts/item min={min(counts)} max={max(counts)} avg={np.mean(counts):.1f}")


def _save_pt(data, path, label):
    import torch

    torch.save(data, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Saved {label}: {path} ({size_mb:.1f} MB)")


def extract(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    if not args.sem_ids:
        raise SystemExit('--sem_ids is required. Set it in the launcher script or pass it on the CLI.')
    sem_ids_path = os.path.abspath(args.sem_ids)
    checkpoint_path = os.path.abspath(args.checkpoint)

    suffix = '' if args.split == 'train' else f'_{args.split}'
    tag = f'{args.dataset}_{args.category}{args.tag_suffix}{suffix}'

    if args.output_dir:
        hs_dir = logits_dir = enc_dir = os.path.abspath(args.output_dir)
    else:
        if not args.graph_dir:
            raise SystemExit('--graph_dir is required when --output_dir is not set.')
        graph_dir = os.path.abspath(args.graph_dir)
        hs_dir = os.path.join(graph_dir, 'hidden_states', tag)
        logits_dir = os.path.join(graph_dir, 'logits', tag)
        enc_dir = hs_dir

    modes = args.mode
    need_hs = 'hidden_states' in modes or 'all' in modes
    need_logits = 'logits' in modes or 'all' in modes
    need_enc = 'encoder_sequences' in modes or 'all' in modes
    need_decoder = need_hs or need_logits

    for d in {hs_dir, logits_dir, enc_dir}:
        os.makedirs(d, exist_ok=True)

    print(f"Initializing Pipeline ...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Dataset: {args.dataset}/{args.category}  split={args.split}")
    print(f"  Modes: {modes}")

    _ensure_import_paths()
    from genrec.pipeline import Pipeline

    pipeline = Pipeline(
        model_name='TIGER',
        dataset_name=args.dataset,
        checkpoint_path=checkpoint_path,
        config_dict={
            'category': args.category,
            'custom_sem_ids_path': sem_ids_path,
        },
    )
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    device = pipeline.config['device']
    model.eval()
    model.to(device)

    tokens_to_item_idx, item_idx_to_info = build_reverse_mapping(
        tokenizer, pipeline.raw_dataset)
    print(f"  Mapped {len(tokens_to_item_idx)} items")

    for d in {hs_dir, logits_dir}:
        _save_item_mapping(item_idx_to_info, d)

    split_dataset = pipeline.tokenized_datasets[args.split]
    loader = DataLoader(
        split_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=tokenizer.collate_fn[args.split],
    )
    n_digit = tokenizer.n_digit

    print(f"\n  Samples: {len(split_dataset)}  batch_size={args.batch_size}  "
          f"test={args.test}")

    hs_per_item = defaultdict(list) if need_hs else None
    logits_per_item = defaultdict(list) if need_logits else None
    enc_per_item = defaultdict(list) if need_enc else None
    total = 0
    unmapped = 0
    seq_len_stats = []

    if need_decoder:
        print(f"\nPass 1: decoder forward (hs={need_hs}, logits={need_logits}) ...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc="Decoder fwd")):
                if args.test and batch_idx >= 2:
                    break

                batch_gpu = {k: v.to(device) for k, v in batch.items()}
                outputs = model.t5(
                    input_ids=batch_gpu['input_ids'],
                    attention_mask=batch_gpu['attention_mask'],
                    labels=batch_gpu['labels'],
                    output_hidden_states=need_hs,
                )

                labels_cpu = batch_gpu['labels'].cpu()

                hs_batch = None
                if need_hs:
                    hs_batch = outputs.decoder_hidden_states[-1][:, :n_digit, :].cpu()

                logits_batch = None
                if need_logits:
                    logits_batch = outputs.logits[:, :n_digit, :].cpu()

                for i in range(labels_cpu.shape[0]):
                    tok = tuple(labels_cpu[i, :n_digit].tolist())
                    idx = tokens_to_item_idx.get(tok)
                    if idx is None:
                        unmapped += 1
                        continue
                    total += 1
                    if hs_batch is not None:
                        hs_per_item[idx].append(hs_batch[i])
                    if logits_batch is not None:
                        logits_per_item[idx].append(logits_batch[i])

        print(f"  Processed: {total}  unmapped: {unmapped}")

    if need_enc:
        enc_label = "raw embeddings" if args.raw_embeddings else "encoder-only"
        print(f"\nPass 2: {enc_label} (encoder_sequences) ...")
        enc_total = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader, desc=enc_label)):
                if args.test and batch_idx >= 2:
                    break

                batch_gpu = {k: v.to(device) for k, v in batch.items()}
                if args.raw_embeddings:
                    enc_hs = model.t5.shared(batch_gpu['input_ids'])
                else:
                    enc_out = model.t5.get_encoder()(
                        input_ids=batch_gpu['input_ids'],
                        attention_mask=batch_gpu['attention_mask'],
                        return_dict=True,
                    )
                    enc_hs = enc_out.last_hidden_state
                attn_mask = batch_gpu['attention_mask']
                labels_cpu = batch_gpu['labels'].cpu()

                for i in range(enc_hs.shape[0]):
                    tok = tuple(labels_cpu[i, :n_digit].tolist())
                    idx = tokens_to_item_idx.get(tok)
                    if idx is None:
                        continue
                    valid_len = int(attn_mask[i].sum().item())
                    enc_per_item[idx].append(enc_hs[i, :valid_len, :].cpu())
                    seq_len_stats.append(valid_len)
                    enc_total += 1

        print(f"  Encoder samples: {enc_total}")
        if seq_len_stats:
            print(f"  Seq len: min={min(seq_len_stats)} max={max(seq_len_stats)} "
                  f"mean={np.mean(seq_len_stats):.1f} median={np.median(seq_len_stats):.0f}")

    print(f"\nSaving results ...")

    if hs_per_item:
        hs_dict = {k: torch.stack(v) for k, v in
                   tqdm(hs_per_item.items(), desc="Stack hs")}
        _print_context_stats(hs_dict, "hidden_states")
        _save_pt(hs_dict, os.path.join(hs_dir, 'hidden_states.pt'), 'hidden_states')
        del hs_dict

    if logits_per_item:
        logits_dict = {k: torch.stack(v) for k, v in
                       tqdm(logits_per_item.items(), desc="Stack logits")}
        _print_context_stats(logits_dict, "logits")
        _save_pt(logits_dict, os.path.join(logits_dir, 'logits.pt'), 'logits')
        del logits_dict

    if enc_per_item:
        enc_dict = dict(enc_per_item)
        _print_context_stats(enc_dict, "encoder_sequences")
        _save_pt(enc_dict, os.path.join(enc_dir, 'encoder_sequences.pt'), 'encoder_sequences')
        del enc_dict

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='Extract TIGER features for SID-MLP training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to TIGER model checkpoint (.pth)')
    parser.add_argument('--dataset', type=str, default='AmazonReviews2023')
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--mode', type=str, nargs='+',
                        default=['all'],
                        choices=['hidden_states', 'logits', 'encoder_sequences', 'all'],
                        help='What to extract (default: all)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory')
    parser.add_argument('--sem_ids', type=str, required=True,
                        help='Path to semantic IDs')
    parser.add_argument('--graph_dir', type=str, default=os.environ.get('SID_MLP_GRAPH_DIR'),
                        help='Base directory containing hidden_states/ and logits/')
    parser.add_argument('--tag_suffix', type=str, default='',
                        help='Suffix appended to dataset tag (e.g. "_rqvae")')
    parser.add_argument('--test', action='store_true',
                        help='Test mode: only process 2 batches')
    parser.add_argument('--raw_embeddings', action='store_true',
                        help='Save raw token embeddings instead of encoder output. '
                             'For encoder distillation: skip encoder, use embed table only.')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
