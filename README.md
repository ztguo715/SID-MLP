# SID-MLP

This repository provides the implementation for **MLPs are Efficient Distilled Generative Recommenders**.

In this work, we address the high inference latency bottleneck of autoregressive decoding in Generative Recommendation (GR) models using Semantic IDs. We introduce SID-MLP, a lightweight distillation framework that eliminates dense attention overhead by distilling heavy autoregressive teachers into position-specific MLP heads. We further propose SID-MLP++, extending this framework to replace the Transformer encoder entirely, demonstrating an 8.74$\times$ inference acceleration while preserving plug-and-play compatibility and teacher-level accuracy.

## Resources

The Amazon Reviews 2023 benchmark is loaded through Hugging Face Datasets from
[`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023).
The first run downloads and caches the selected category under `CACHE_DIR`.

SID-MLP distillation also needs a trained TIGER checkpoint and the matching
semantic ID file. Either train TIGER locally with this repo, or place released
teacher assets under `MODEL_DIR` with this layout:

```text
<MODEL_DIR>/
  TIGER-AmazonReviews2023-category_<category>.pth
  semantic_ids/
    AmazonReviews2023-<category>_sentence-t5-base_256,256,256,256.sem_ids
```

All data, checkpoint, feature, and output paths are passed through launcher
environment variables or CLI flags.

## Installation

```bash
conda env create -f SID-MLP/environment.yml
conda activate TIGER_env
```

For an existing Python environment:

```bash
pip install -r SID-MLP/requirements.txt
```

Run commands from the parent project root:

```bash
export PROJECT_DIR=<parent-project-root>
export SID_MLP_DIR=$PROJECT_DIR/SID-MLP
export GRAPH_DIR=<feature-output-root>
export MODEL_DIR=<tiger-checkpoint-root>
export CACHE_DIR=<amazon-reviews-cache-root>
export PYTHONPATH=$SID_MLP_DIR:$PROJECT_DIR
```

The repository directory can be named `SID-MLP`; the importable Python package
is `sid_mlp`.

## Training

Train a TIGER teacher:

```bash
CACHE_DIR=$CACHE_DIR CKPT_DIR=<teacher-output-root> \
  bash SID-MLP/scripts/train_tiger.sh <category> <gpu>
```

The local `genrec/` copy only includes the TIGER teacher and AmazonReviews2023
data pipeline. Teacher outputs are controlled by `CACHE_DIR`, `CKPT_DIR`,
`LOG_DIR`, and `TENSORBOARD_DIR`.

Extract frozen TIGER features:

```bash
bash SID-MLP/scripts/extract.sh teacher <category> train <gpu>
bash SID-MLP/scripts/extract.sh teacher <category> val <gpu>
```

Build the valid semantic-ID mask:

```bash
bash SID-MLP/scripts/build_valid_set.sh <dataset_tag>
```

Train SID-MLP:

```bash
bash SID-MLP/scripts/train.sh sidmlp <category> <gpu>
```

Training defaults are in `configs/sidmlp.yaml`. Hyperparameter sweep ranges are
documented as comments in the config files.

## Inference

Run SID-MLP inference:

```bash
bash SID-MLP/scripts/infer.sh sidmlp <category> <gpu> <sid_mlp_ckpt> test
```

Inference defaults are in `configs/infer.yaml`. CUDA inference uses native bf16
by default with TF32 disabled. Set `FP32=1` or pass `--fp32` for fp32 inference.

## SID-MLP++

Train the stage-1 encoder:

```bash
bash SID-MLP/scripts/train.sh sidmlp-pp-stage1 \
  <category> <gpu> <num_layers> <ffn_dim> <tag>
```

Transform raw embeddings and train the stage-2 decoder:

```bash
bash SID-MLP/scripts/extract.sh raw <category> train <gpu>
bash SID-MLP/scripts/extract.sh raw <category> val <gpu>

bash SID-MLP/scripts/extract.sh sidmlp-pp \
  <category> train <gpu> <encoder_ckpt> <num_layers> <ffn_dim> <tag>
bash SID-MLP/scripts/extract.sh sidmlp-pp \
  <category> val <gpu> <encoder_ckpt> <num_layers> <ffn_dim> <tag>

bash SID-MLP/scripts/build_valid_set.sh <transformed_dataset_tag>
bash SID-MLP/scripts/train.sh sidmlp-pp-stage2 \
  <category> <gpu> <encoder_ckpt> <num_layers> <ffn_dim> <tag>
```

Run SID-MLP++ inference:

```bash
bash SID-MLP/scripts/infer.sh sidmlp-pp \
  <category> <gpu> <sid_mlp_pp_ckpt> <encoder_ckpt> <tag> test
```

SID-MLP++ inference requires `--encoder_ckpt` and replaces the TIGER encoder
with the distilled encoder.

## Files

- `sid_mlp/`: SID-MLP and SID-MLP++ implementation.
- `genrec/`: TIGER-only teacher training code.
- `configs/`: training and inference defaults.
- `scripts/`: launchers for extraction, training, valid-mask construction, and
  inference.
