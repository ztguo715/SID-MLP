import argparse

from genrec.utils import get_pipeline, parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate the TIGER teacher.")
    parser.add_argument("--model", default="TIGER", choices=["TIGER"])
    parser.add_argument("--dataset", default="AmazonReviews2023", choices=["AmazonReviews2023"])
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--config_file", default=None)
    args, unknown = parser.parse_known_args()
    return args, parse_command_line_args(unknown)


def main():
    args, config = parse_args()
    pipeline_cls = get_pipeline(args.model)
    pipeline = pipeline_cls(
        model_name=args.model,
        dataset_name=args.dataset,
        checkpoint_path=args.checkpoint_path,
        config_file=args.config_file,
        config_dict=config,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
