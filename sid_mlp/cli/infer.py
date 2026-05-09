"""Run SID-MLP or SID-MLP++ beam-search inference."""

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


def main(argv=None):
    config = _load_config(_find_config_path(argv))
    parser = argparse.ArgumentParser(description="SID-MLP public inference CLI")
    parser.add_argument("--config", type=str, default=None, help="YAML file with inference defaults")
    parser.add_argument("--checkpoint", type=str, default=None, help="TIGER teacher checkpoint")
    parser.add_argument("--dataset", type=str, default="AmazonReviews2023")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--dataset_tag", type=str, default=None)
    parser.add_argument("--lens_path", type=str, default=None, help="SID-MLP decoder checkpoint")
    parser.add_argument("--valid_set", type=str, default=None)
    parser.add_argument("--sem_ids", type=str, default=None)
    parser.add_argument("--graph_dir", type=str, default=os.environ.get("SID_MLP_GRAPH_DIR"))
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="SID-MLP++ encoder checkpoint; required for SID-MLP++ inference",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beam_size", type=int, default=50)
    parser.add_argument("--final_top_n", type=int, default=50)
    parser.add_argument("--d1_temp", type=float, default=1.0)
    parser.add_argument("--d2_temp", type=float, default=1.0)
    parser.add_argument("--d3_temp", type=float, default=1.0)
    parser.add_argument("--d4_temp", type=float, default=1.0)
    parser.add_argument("--alpha_d2", type=float, default=1.0)
    parser.add_argument("--gamma_d3", type=float, default=1.0)
    parser.add_argument("--delta_d4", type=float, default=1.0)
    parser.add_argument("--split", type=str, default=None, choices=["val", "test"])
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--fp32", action="store_true", help="Use fp32 inference instead of default native bf16")

    _apply_config_defaults(parser, config)
    args = parser.parse_args(argv)
    _require(args, "checkpoint", "category", "dataset_tag", "lens_path", "sem_ids")
    from ..inference import run_inference

    run_inference(args)


if __name__ == "__main__":
    main()
