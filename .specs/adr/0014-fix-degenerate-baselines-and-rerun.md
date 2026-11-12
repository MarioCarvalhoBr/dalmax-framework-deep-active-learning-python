# ADR 0014: Fix the seven degenerate paper-1 baselines and re-run them as two campaign parts

- **Status:** Accepted (author decision 2026-11-11); code fixed 2026-11-11. Operational part (Decision 4-5 and the
  amendment: re-run parts, archive script, superseded-leaf guard) **superseded by ADR-0016** on 2026-11-11 before any
  execution: everything is re-run on `DATA/daninhas_full_fixed` into new roots, the 21 groups are back in `paper1`, and
  `dalmax/superseded.py` / `scripts/campaign/archive_superseded_runs.py` were deleted. The code fixes (Decision 1-3) stand.
- **Date:** 2026-11-11

## Context

Paper 1's benchmark (12 classical strategies x `n_query` {10, 50, 100} x seeds 1-3, campaign part `paper1`,
executed 2026-09-30 to 2026-10-05) evaluated seven strategies in a degenerate form
(`.specs/quality/known-issues.md`):

- **KI-41**: `DaninhasModelResNet50` had no dropout layer and `DeepLearning.predict_prob_dropout*` only called
  `clf.train()`, so the `n_drop` passes were identical (BatchNorm in train mode over an unshuffled loader) and
  `LeastConfidenceDropout`, `MarginSamplingDropout`, `EntropySamplingDropout` and `BALDDropout` did not measure an
  MC-dropout posterior; BALD's mutual information was rounding noise and `0 * log(0)` could give NaN.
- **KI-42**: `forward` returned `e1 = relu(embedding_layer(features))`, an untrained random 50-d projection, so
  `KMeansSampling` and `KCenterGreedy` did not use the trained penultimate representation as in DeepAL.
  `KCenterGreedy` also took `np.sqrt` of slightly negative squared distances (NaN rows, which win the `argmax`).
- **KI-45**: `PerSampleAdversarialStrategy.query` ranked with `dis.argsort(kind="stable")`; BIM distances take few
  discrete values, the DANINHAS pool is stored class by class, so `AdversarialBIM` selected "the lowest unlabeled
  indices".

On 2026-11-02 (KI-41/KI-42) and earlier on 2026-11-11 (KI-45) the runs were accepted as executed and disclosed.
On 2026-11-11 the author decided instead to fix all seven and re-run them (7 x 3 x 3 = 63 runs) on Google Colab,
split over two Colab accounts running at the same time. Hard constraint: the other five paper-1 strategies
(RandomSampling, LeastConfidence, MarginSampling, EntropySampling, AdversarialDeepFool), KMH and papers 2/3 are not
re-run, so their behaviour must stay bit-identical on CPU (same RNG stream at model construction, same training,
same selection).

## Decision

We will fix the three defects in code and re-run only the affected 21 groups:

1. **MC dropout (KI-41)**, as in DeepAL (`nets.py`: `F.dropout(e1, training=self.training)`, p = 0.5, between the
   penultimate representation and the final linear layer). `DaninhasModelResNet50(n_classes, dropout_p=0.0)`
   applies `nn.Dropout(dropout_p)` to the 2048-d pooled features before `classifier` when `dropout_p > 0`
   (`nn.Identity` otherwise). The value comes from a new optional params-JSON block in the dataset entry,
   `"model_args": {"dropout_p": <float in [0, 1)>}` (`dalmax/config/schema.py::ModelArgs`, parsed and validated by
   `dalmax/config/loader.py`, unknown keys rejected), threaded by `dalmax/models/registry.py::get_network` into
   `DeepLearning(..., model_kwargs=...)`, which builds `net(n_classes, **model_kwargs)` every round, and recorded
   in the checkpoint (`model_kwargs`, optional key: older checkpoints load as `{}`). An absent block passes nothing
   (resolved config `"model_args": {"dropout_p": null}`), i.e. the model's own default: no dropout for the DANINHAS
   ResNet-50, the built-in DeepAL p = 0.5 of the CIFAR10 CNN. **Only the four MC-dropout strategies train with the
   dropout layer** (`files_config/campaign/params_paper1_mc_dropout.json` = `params_paper1.json` + `dropout_p` 0.5),
   so their round-0 model differs from the other strategies' round-0 model; this is inherent to MC dropout (the
   network that is sampled must be trained with dropout) and is stated wherever they are compared.
   `predict_prob_dropout*` now run under `dalmax/models/mc_dropout.py::mc_dropout_mode`: model in eval mode
   (BatchNorm uses its running statistics and no longer updates them), only the dropout modules in train mode,
   every module's mode restored afterwards, and a fail-fast `MCDropoutError` naming KI-41 when the model has no
   dropout module with p > 0. They use their `n_drop` argument (the callers always passed `params['n_drop']`).
   `BALDDropout` and `EntropySamplingDropout` compute `p log p` as `torch.xlogy(p, p)`; `EntropySampling` (not
   re-run) is untouched. The CIFAR10 CNN's functional dropout becomes an equivalent `nn.Dropout` module so that MC
   prediction keeps working for it.
2. **Trained embeddings (KI-42)**: `forward` returns the 2048-d pooled features that feed the classifier as `e1`
   (`get_embedding_dim() == 2048`). `embedding_layer = nn.Linear(2048, 50)` is **kept, constructed in the same
   position and unused**: its construction draws from the torch RNG before the classifier's initialisation, and its
   keys are in every saved checkpoint, so removing it would change every strategy's training and break checkpoint
   loading. Only `get_embeddings` consumes `e1` (the adversarial strategies unpack and ignore it). `KCenterGreedy`
   clamps the squared distances at 0 before the square root; its algorithm is otherwise unchanged. `KMeansSampling`
   is unchanged (its `KMeans(n_clusters=n)` without `random_state` draws from NumPy's global RNG, which
   `seed_everything` seeds, so it is reproducible).
3. **Random tie-break (KI-45)** in the shared adversarial base: `rank_by_distance(dis, rng)` =
   `np.lexsort((rng.permutation(len(dis)), dis))`, with `rng = np.random.default_rng(derive_seed(config.seed,
   "adversarial_ties"))` created once per run by `build_strategy` (`LEGACY_STRATEGY_KWARGS`) and injected through
   the `AdversarialBIM` constructor; it advances across rounds. `inf` still ranks last; without ties the order equals
   the old stable argsort. **`AdversarialDeepFool` is not given the generator** (`rng=None`, the base then keeps the
   stable argsort, i.e. exactly its pre-fix code path). Reason, found while implementing: DeepFool distances are
   continuous, but 1,269 of the 8,086 DANINHAS training files are byte-identical copies of another file (588 groups, never across classes), and duplicates get exactly equal distances. A random tie-break would therefore change
   DeepFool's selection whenever a duplicate pair straddles the batch boundary, which breaks the constraint that the
   five unchanged strategies keep their behaviour; keeping the pool order there is harmless, because the tied samples
   are the same image with the same label (only which copy is picked differs, there is no class bias). Switching DeepFool
   to the random tie-break is one registry entry if ever wanted (it would then need a re-run too). The duplicates themselves, including 171 images
   present in both train and test, are recorded as KI-46 (open, author decision).
4. **Re-run parts**: the 21 groups `p1/<Strategy>/nq{10,50,100}` keep their ids, `dir_results` and table rows
   (results land in the same leaf paths) and move from part `paper1` to two parts balanced by the GPU time of the
   first execution (BIM about 2,300 s per run, MC dropout about 230 s, KMeans/KCenter about 90 s): `paper1_rerun_a`
   (AdversarialBIM nq100 and nq50, KMeansSampling and KCenterGreedy at all budgets; 24 runs, about 4.3 h) and
   `paper1_rerun_b` (AdversarialBIM nq10 and the four MC-dropout strategies; 39 runs, about 4.2 h), one per Colab
   account. `paper1` becomes 54 runs / 18 groups; the totals stay 222 runs / 74 groups.
5. **Archive of the superseded runs**: `scripts/campaign/archive_superseded_runs.py`
   (`make campaign-archive-superseded [DRY=1|CHECK=1]`) moves every leaf of the two parts whose
   `run_metadata.json` `started_at` precedes the fix cutoff `SUPERSEDED_BEFORE = 2026-11-11T19:12:28Z` to
   `results/old_results_broken/<path below results/>`, logging each move in
   `results/old_results_broken/ARCHIVE_LOG.tsv`; it never deletes or overwrites, never moves a leaf started at or
   after the cutoff, and `--check` fails while a stale leaf is still in the live tree. This move is the one
   sanctioned exception to "`results/` is append-only", an explicit author instruction to preserve the old runs.

## Consequences

- Positive: the seven baselines measure what their names say; the old runs stay available for the record
  (`results/old_results_broken/`); tables, report and thesis keep reading the same leaf paths and table rows.
- Behaviour preservation (verified on CPU): with `model_args` absent the ResNet-50 has the same parameters,
  `state_dict` keys, RNG draws at construction and logits as before (`tests/test_mc_dropout_and_embeddings.py`
  against a frozen copy of the pre-fix class); the golden fixtures `tests/golden/random_sampling_micro_seed1.json`
  and `ssrae_kmeans_micro_seed1.json` were not regenerated and are reproduced exactly (initial set, round-1 query,
  metrics, checkpoint re-prediction); micro CPU runs (seed 1, nq 5, NIL 10, NR 1) of RandomSampling, LeastConfidence,
  MarginSampling, EntropySampling and KMH gave identical initial sets, query indices and bit-identical metrics before
  (commit 919a896) and after the change. AdversarialDeepFool was not compared end to end (one micro CPU run takes
  hours); it is built exactly as before (no generator) and its ranking is the same `argsort(kind="stable")` call
  (`tests/test_adversarial_strategies.py::test_deepfool_keeps_the_pool_order_tie_break_of_its_executed_runs`); its
  `cal_dis` is untouched.
- The re-run executes on a later Colab runtime image than the five unchanged strategies (KI-43): paper-1 tables mix
  images; compare seed means and treat differences within the run-to-run variation as ties.
- The MC-dropout strategies' round 0 now differs from the others' (they train a network with dropout); paper 1 states
  this when it discusses round 0.
- Every `run_metadata.json` written from now on carries `config.dataset.model_args`; `effective_signature` includes
  it, so the MC-dropout groups' signature changed (they are not aliases of anything).
- Follow-up: execute the two parts (`COLAB_RUNBOOK.md` section 6d), archive, merge, `make campaign-verify` (expect
  `OK=222`), `make campaign-report`, then update paper 1's tables and the KI-41/KI-42/KI-45 disclosures in the thesis.

## Amendment (2026-11-11, safeguards against keeping or shipping the superseded leaves)

Two reviews found that the only protection against launching the re-run parts over the old leaves (skip-existing would
keep them as complete, `verify` would count them `OK`) or zipping old leaves was an operator step. With the cloud
credits nearly exhausted, that mistake must be impossible, not just documented:

1. **One definition** in `dalmax/superseded.py` (stdlib only; imported by `dalmax/campaign.py`, by
   `scripts/campaign/archive_superseded_runs.py` and, for the part names, by `scripts/campaign/build_manifest.py`):
   `RERUN_PARTS`, `SUPERSEDED_BEFORE` and `classify_leaf`. A leaf of a re-run part is **superseded** when its
   `run_metadata.json` (a) started before the cutoff, OR (b) lacks `config.dataset.model_args`. Rule (b) holds because
   `run_metadata.json` stores `to_dict(config)` (`dataclasses.asdict`), and since the fix `DatasetConfig.model_args`
   always exists (`{"dropout_p": null}` when the params JSON has no block); none of the 63 first-execution leaves under
   `results/campaign/paper1/` has the key (checked read-only 2026-11-11). Without a readable `run_metadata.json` (or
   with model_args but no valid `started_at`) a leaf is **interrupted** if it has no `results.json` (not superseded; the
   engine re-runs it, as `job_done` already treats any incomplete leaf) and **unreadable** if it has one (blocking,
   resolved by hand, `COLAB_RUNBOOK.md` section 6d). The archive script keeps its rules (never delete, never overwrite,
   never move a current, interrupted or unreadable leaf) and now also rejects `..` components in `dir_results`.
2. **Engine guard**: `campaign run` (real, `--dry-run`, `--no-skip-existing`) raises `SupersededLeavesError` before
   launching or logging anything when a selected job of a re-run part has a superseded or unreadable leaf (every
   matching leaf is checked): exit 3, the count, the first five leaves and the archive commands. `campaign verify`
   reports such jobs as `STALE` (non-OK, exit 1; `Totals: ... STALE=n`), so `OK=24` / `OK=39` / `OK=222` can only mean
   runs of the fixed code. Jobs of other parts are never inspected. On the local tree (2026-11-11, before archiving):
   `OK=159 STALE=63`, and `run --part paper1_rerun_a --dry-run` refuses with 24 leaves.
3. **Notebook**: the archive check cell and a new gate cell right before the run cell call the check with
   `subprocess.run(..., check=True)`, so "Run all" stops there; the gate also asserts that `campaign-list` reports 24 runs
   / 8 groups (`paper1_rerun_a`) or 39 / 13 (`paper1_rerun_b`), which the Parameters cell prints in a banner. The zip
   cell refuses unless the check passes and verify reports every run of the part `OK` (explicit `ZIP_PARTIAL` opt-out),
   checks the checkpoint count, and adds the account's `campaign.log` / `failures.log` as
   `results/campaign/{campaign,failures}_<PART>.log` so the measured times survive the merge.
4. Not done: failing at startup (in `build_strategy`) for an MC-dropout strategy configured without dropout. A
   config-level check is not possible (`dropout_p: null` means the model's own default, 0.0 for the DANINHAS ResNet-50
   but 0.5 for the CIFAR10 CNN) and building a probe model would draw from the torch RNG, i.e. change the reviewed runs.
   The existing fail-fast `MCDropoutError` at the first query remains the guard.

No query strategy, model, seeding or config behaviour changed; the manifests are unchanged (`build_manifest.py --check`).
