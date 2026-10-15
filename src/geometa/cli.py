import argparse
import json
from pathlib import Path
import yaml
from .experiment import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run reproducible meta-learning and active regression experiments")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse own saved training checkpoint; regenerate evaluation")
    parser.add_argument("--seed", type=int, help="Override to one seed, useful for Slurm arrays")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.seed is not None: config["seeds"]=[args.seed]
    print(json.dumps({"config":config, "output":args.output}), flush=True)
    run_experiment(config, args.output, args.resume)


if __name__ == "__main__":
    main()

