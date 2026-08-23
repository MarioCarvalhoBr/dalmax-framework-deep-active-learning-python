# Use case: add a new query strategy

Derived from how `SSRAEKmeansSampling`/`SSRAEKmeansHCSampling` are actually
wired today (verified by reading `core/query_strategies/__init__.py`,
`utils/orchestrator.py`, `demo.py`, and the strategy classes themselves).
This is the exhaustive list of files that currently must change — the
if/elif registry pattern is a known issue (`known-issues.md`, owned by
another batch) that the refactor plan intends to replace, but until that
refactor lands, all of these steps are required or the strategy will be
unreachable/broken.

## 1. Implement the strategy class

Create `core/query_strategies/<new_strategy_file>.py`, subclassing
`Strategy` (`core/query_strategies/strategy.py`) or, if it needs the
existing SSL hierarchical machinery, `SSLStrategy`
(`core/query_strategies/ssl_ssrae_sampling.py`). At minimum implement
`query(self, n)` returning an array of `n` unlabeled sample indices (by
`img_id`, matching the keys used in `dataset.features_dict` for
embedding-based strategies, or by pool index for uncertainty-based ones —
follow the existing convention of the closest sibling strategy).

Constructor signature must match the existing pattern:
`def __init__(self, dataset, net, logger): super().__init__(dataset, net, logger)`
— strategies do **not** receive `params` via the constructor; `demo.py`
injects it after construction via `setattr(strategy, "params", params)`
(a known-issue anti-pattern, not something a new strategy should try to
work around by changing the constructor signature, since `get_strategy`
below instantiates with exactly `(dataset, net, logger)`).

## 2. Register the class export

Add `from .<new_strategy_file> import <NewStrategyClass>` to
`core/query_strategies/__init__.py`.

## 3. Register in `utils/orchestrator.py`

`get_strategy(name)` is an if/elif chain (`utils/orchestrator.py:45+`) —
add:
```python
elif name == "<NewStrategyClass>":
    return <NewStrategyClass>
```
(it returns the **class**, not an instance — `demo.py` calls
`get_strategy(args.strategy_name)(dataset, net, logger)`).

## 4. Register in `demo.py` CLI choices

Add `"<NewStrategyClass>"` to the `choices=[...]` list of
`parser.add_argument('--strategy_name', ...)` in `demo.py`
(lines ~267–285) — **without this, the strategy cannot be selected from the
CLI at all**, even if steps 1–3 are done correctly; `argparse` will reject
the value before `main()` ever runs.

## 5. Params JSON, if the strategy needs new config

If the strategy needs new hyperparameters (like `config_kmh` for the
hierarchical strategies), add the key(s) under the relevant dataset section
of every params JSON file that will be used to run it
(`params_df_gpu_0.json`, `params_df_gpu_1.json`, and any others in active
use — `params_dnf.json` is referenced by `scripts/run_pipline.sh` but not
present in the repo, see `experiments/experimental-protocol.md`). Access it
in the strategy via `self.params[<dataset_name>][<key>]` — but **do not**
hardcode the dataset name key the way
`core/query_strategies/ssl_ssrae_sampling.py:66` does
(`self.params['DANINHAS']['config_kmh']`, unconditional regardless of
`--dataset_name`); use the actual `dataset_name` the strategy is running
against if it needs to be dataset-agnostic (the current SSL strategies are
not, and cannot run against CIFAR10 as a result — see `known-issues.md`).

## 6. Feature/embedding caching, if applicable

If the strategy needs precomputed image embeddings (like SSRAE/VCTex),
wire it into `Data.initialize_labels`
(`utils/data.py:211-229`) — currently an if/elif on `self.strategy_name`
that calls `create_feature_maps_ssrae`/`create_feature_maps_vctex`. Follow
the reproducibility rule in `research-rules/reproducibility.md`: any new
cache must be explicitly keyed (dataset, extractor, hyperparameters), not
appended to the existing unkeyed `features_dict_*.pkl` pattern.

## 7. Tests

Add or extend `tests/test_registry.py` (owned by the quality-harness batch)
so `utils.orchestrator.get_strategy("<NewStrategyClass>")` is asserted to
resolve without error — this is the automated version of step 3/4's
correctness check. If the strategy has non-trivial pure-Python logic
(e.g. an embedding slicing rule), add a small, fast, CPU-only unit test
following the pattern of `tests/test_ssrae_embedding_layout.py`.

## 8. Update specs

Update this file if the checklist itself changes (e.g. once the registry
becomes a real registry pattern instead of if/elif, delete steps 3–4 and
describe the registration decorator/entry instead), and update
`experiments/experimental-protocol.md`'s "Query strategies exercised"
section if the new strategy is added to any run script's sweep.

## Reference: current full strategy list (for cross-checking step 4)

`RandomSampling`, `LeastConfidence`, `MarginSampling`, `EntropySampling`,
`LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout`,
`KMeansSampling`, `KCenterGreedy`, `BALDDropout`, `AdversarialBIM`,
`AdversarialDeepFool`, `SSRAEKmeansSampling`, `VCTexKmeansSampling`,
`SSRAEKmeansHCSampling`, `VCTexKmeansHCSampling`.
