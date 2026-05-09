"""Public training entrypoint for SID-MLP and SID-MLP++."""

import argparse
import ast
import os
import sys

import yaml


def _find_config_path(argv):
    argv = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(argv):
        if token == "--config":
            if index + 1 >= len(argv):
                raise SystemExit("--config requires a path")
            return argv[index + 1]
        if token.startswith("--config="):
            return token.split("=", 1)[1]
    return None


def _load_config(path):
    if not path:
        return {}
    with open(path, "r") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise SystemExit(f"Config must be a YAML mapping: {path}")
    return {key: _coerce_config_value(value) for key, value in loaded.items()}


def _coerce_config_value(value):
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _apply_config_defaults(parser, config):
    if not config:
        return
    valid_dests = {action.dest for action in parser._actions}
    parser.set_defaults(**{
        key: value
        for key, value in config.items()
        if key in valid_dests and value is not None
    })


def _require(args, *names):
    missing = [f"--{name}" for name in names if not getattr(args, name, None)]
    if missing:
        raise SystemExit(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Pass them on the command line or in --config."
        )


def _add_decoder_args(parser):
    parser.add_argument("--config", type=str, default=None, help="YAML file with training defaults")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="AmazonReviews2023")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--dataset_tag", type=str, default=None)
    parser.add_argument("--val_dataset_tag", type=str, default=None)
    parser.add_argument("--valid_set", type=str, default=None)
    parser.add_argument("--sem_ids", type=str, default=None)
    parser.add_argument("--graph_dir", type=str, default=os.environ.get("SID_MLP_GRAPH_DIR"))
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--attn_dim", type=int, default=384)
    parser.add_argument("--ffn_dim", type=int, default=1024)
    parser.add_argument("--head_hidden", type=int, default=512)
    parser.add_argument("--head_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--loss", type=str.lower, choices=["kl", "ce", "combined"], default="combined")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--d4_teacher", type=str.lower, choices=["kl", "ce", "combined"], default="combined")
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--val_batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--final_top_n", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--storage_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--test", action="store_true")


def _add_encoder_args(parser):
    parser.add_argument("--config", type=str, default=None, help="YAML file with training defaults")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--raw_dir", type=str, default=None)
    parser.add_argument("--teacher_dir", type=str, default=None)
    parser.add_argument("--val_raw_dir", type=str, default=None)
    parser.add_argument("--val_teacher_dir", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--encoder_ckpt", type=str, default=None)
    parser.add_argument("--extract_out", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--ffn_dim", type=int, default=2048)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--gpu_resident", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--storage_fp16", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)


def _run_encoder(args):
    from ..pretrain_encoder import extract_transformed, train_encoder

    _require(args, "raw_dir")
    if args.extract:
        _require(args, "encoder_ckpt", "extract_out")
        extract_transformed(args)
    else:
        _require(args, "teacher_dir", "output")
        train_encoder(args)


def _run_decoder(args, *, require_encoder):
    _require(args, "checkpoint", "category", "dataset_tag", "val_dataset_tag", "graph_dir", "sem_ids")
    if require_encoder and not args.encoder_ckpt:
        raise SystemExit("SID-MLP++ stage2 requires --encoder_ckpt")
    from ..train_decoder import train_sidmlp

    train_sidmlp(args)


def main(argv=None):
    config = _load_config(_find_config_path(argv))
    parser = argparse.ArgumentParser(description="SID-MLP public training CLI")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    sidmlp = subparsers.add_parser("sidmlp", help="train SID-MLP decoder")
    _add_decoder_args(sidmlp)
    sidmlp.set_defaults(_runner=lambda args: _run_decoder(args, require_encoder=False))

    stage1 = subparsers.add_parser("sidmlp-pp-stage1", help="train/apply SID-MLP++ encoder")
    _add_encoder_args(stage1)
    stage1.set_defaults(_runner=_run_encoder)

    stage2 = subparsers.add_parser("sidmlp-pp-stage2", help="train SID-MLP decoder on SID-MLP++ states")
    _add_decoder_args(stage2)
    stage2.set_defaults(_runner=lambda args: _run_decoder(args, require_encoder=True))

    for subparser in (sidmlp, stage1, stage2):
        _apply_config_defaults(subparser, config)

    args = parser.parse_args(argv)
    if hasattr(args, "loss") and isinstance(args.loss, str):
        args.loss = args.loss.lower()
    if hasattr(args, "d4_teacher") and isinstance(args.d4_teacher, str):
        args.d4_teacher = args.d4_teacher.lower()
    args._runner(args)


if __name__ == "__main__":
    main()
