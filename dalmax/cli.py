"""Thin CLI: argparse identical to the pre-Phase-2 historical `demo.py`, plus
the new `--device` and `--embedding_variant` flags and the
`RepresentationStrategy` `--strategy_name` choice — everything else (config
loading, the round loop, reporting) is delegated to `dalmax.config.loader`,
`dalmax.experiment.runner`, and `dalmax.experiment.reporter`
(`.specs/architecture/target-architecture.md` §10). `trainer.py` (renamed
from the historical `demo.py` on 2026-08-23, see ADR 0006) at the repo root
is a thin shim that calls `main()` here, so existing shell scripts
(`scripts/benchmark/run_pipe_gpu_*.sh`) keep working unchanged.

The list of valid `--strategy_name` values here is the single source of
truth `tests/test_registry.py`/`tests/test_strategy_registry.py` parse via
`ast` (see those tests' docstrings for why they never `import dalmax.cli`
directly: importing it eagerly creates a `results/logs/` log file via
`dalmax.logging_utils`, same side effect the historical `demo.py` always had).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

from dalmax.config.loader import load_experiment_config, to_dict
from dalmax.config.schema import ExperimentConfig
from dalmax.experiment.reporter import write_report
from dalmax.experiment.runner import ExperimentRunner
from dalmax.logging_utils import get_logger, get_path_logger


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_results", type=str, default="results/dalmax", help="Results directory")
    parser.add_argument(
        "--params_json", type=str, default="params.json", help="Params JSON file. See example in README.md"
    )

    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--n_init_labeled", type=int, default=100, help="number of init labeled samples")
    parser.add_argument("--n_query", type=int, default=10, help="number of queries per round")
    parser.add_argument("--n_round", type=int, default=10, help="number of rounds")
    parser.add_argument(
        "--dataset_name", type=str, default="CIFAR10", choices=["CIFAR10", "DANINHAS"], help="dataset"
    )
    parser.add_argument(
        "--strategy_name",
        type=str,
        default="RandomSampling",
        choices=[
            "RandomSampling",
            "LeastConfidence",
            "MarginSampling",
            "EntropySampling",
            "LeastConfidenceDropout",
            "MarginSamplingDropout",
            "EntropySamplingDropout",
            "KMeansSampling",
            "KCenterGreedy",
            "BALDDropout",
            "AdversarialBIM",
            "AdversarialDeepFool",
            "SSRAEKmeansSampling",
            "VCTexKmeansSampling",
            "SSRAEKmeansHCSampling",
            "VCTexKmeansHCSampling",
            "RepresentationStrategy",
        ],
        help="query strategy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="compute device ('auto' resolves to 'cuda' iff available, else 'cpu')",
    )
    parser.add_argument(
        "--embedding_variant",
        type=str,
        default=None,
        choices=["full", "spatial", "spectral"],
        help=(
            "override the embedding_variant from params_json's 'embedding' block "
            "(SSRAE only; see dalmax/embeddings/variants.py). Default: use params_json as-is."
        ),
    )
    return parser


def _apply_embedding_variant_override(config: ExperimentConfig, variant: str | None) -> ExperimentConfig:
    """Return `config` with `dataset.embedding.variant` overridden by `--embedding_variant`,
    or `config` unchanged if `variant is None` (the default: no override)."""
    if variant is None:
        return config
    new_embedding = replace(config.dataset.embedding, variant=variant)
    new_dataset = replace(config.dataset, embedding=new_embedding)
    return replace(config, dataset=new_dataset)


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logger = get_logger()
    path_logger = get_path_logger()

    logger.warning("==========================================================================>")
    logger.warning("DalMax - Training the model with PyTorch...")
    logger.warning("==========================================================================>")
    logger.warning("ARGUMENTS (cli) AND PARAMETERS (json)")
    logger.warning("--------------------------------------------------------------------------")
    logger.warning("ARGS: " + json.dumps(vars(args), indent=4))
    logger.warning("--------------------------------------------------------------------------")

    config = load_experiment_config(
        args.params_json,
        args.dataset_name,
        strategy_name=args.strategy_name,
        seed=args.seed,
        n_init_labeled=args.n_init_labeled,
        n_query=args.n_query,
        n_round=args.n_round,
        dir_results=args.dir_results,
        device=args.device,
    )
    config = _apply_embedding_variant_override(config, args.embedding_variant)

    logger.warning("CONFIG: " + json.dumps(to_dict(config), indent=4))
    logger.warning("--------------------------------------------------------------------------")

    runner = ExperimentRunner(config, logger)
    result = runner.run()
    write_report(result, path_logger)


if __name__ == "__main__":
    main()
