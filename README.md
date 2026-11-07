# RiskFuse-VLM

Selective mitigation of attribute and relation hallucinations in frozen
vision–language models, with image-level calibration, visual grounding,
self-verification and an availability-aware corpus prior.

This repository contains the `hmqr` implementation and the reproducibility
materials prepared for the manuscript **Image-level calibration for selective
mitigation of attribute and relation hallucinations in vision–language models**.
It includes a small CPU-only reproduction that requires neither model weights
nor benchmark images.

## Reproduce the extraction table

Use Python 3.10 or newer and run these commands from the repository root:

```text
python -m pip install -r requirements-reproduction.txt
python scripts/17_reproduce_extraction_table.py --config configs/default.yaml
```

The script rebuilds the extraction count-and-score table from the final
human-reviewed reference actions and one-to-one prediction matches. It does
not read stored summary counts or the reference-result CSVs.

Outputs:

```text
results/reproduced_extraction/revision_extraction_audit.csv
results/reproduced_extraction/reproduction_manifest.json
```

Expected pooled results:

| Descriptions | Shared images | Matches | Predictions | Reference propositions |
|---:|---:|---:|---:|---:|
| 200 | 50 | 469 | 942 | 3674 |

Pooled precision is **49.79%**, recall **12.77%**, and F1 **20.32%**.
The CSV also contains per-backbone/type and pooled-type rows. This entry point
reconstructs counts and point estimates; it does not recompute bootstrap
intervals or image-truth labels.

The reference was **AI-assisted and reviewed by one human**. It evaluates
whether a proposition is expressed in a caption. It is not independent or
blinded annotation and does not establish whether the image supports that
proposition. See [annotation provenance](docs/ANNOTATION.md).

The reconstruction was executed in a fresh isolated environment during the
manuscript revision. This GitHub directory was assembled afterward; no new
software tests or model experiments were run to prepare this upload.

## What is included

```text
configs/default.yaml                single configuration source
src/hmqr/                           installable implementation
scripts/00_... through 17_...        main pipeline and review entry points
scripts/18_unpack_supplement_snapshot.py
                                    unpack the released supplement inputs
run_all.py                          optional full-pipeline runner
analyse_supplement.py               matched supplementary policy analysis
supplement_worker.py                supplementary GPU signal/feature collection
supplement_generative.py             frozen-policy caption application
supplement_latency.py               fresh native-path timing
supplement_common.py                configuration and workspace resolution
reproduction_data/extraction/       source tasks and final human judgements
reproduction_data/supplement_raw.zip
                                    fixed cohort, raw scores and probe features
reference_results/                  reported outputs, separate from runtime data
docs/                               protocol, data and environment notes
LICENSE                             existing repository MIT license
release_manifest.json               file sizes, hashes and packaging provenance
```

Weights, benchmark image collections, private review workbooks, chat history,
SSH helpers, connection details and machine-specific caches are not included.

## Scope of the released evidence

| Evidence | Scope |
|---|---|
| Historical diagnostics and detection | Frozen aggregates from four backbones; archived separately |
| Textual extraction audit | 200 descriptions, 50 shared images, four backbones |
| New AMBER supplement | LLaVA-1.5; count/state attributes and contact relations; 900 prespecified image–family records |
| Matched policies | 12 policies; 100, 100 and 98 usable test records across the three families |
| Sensitivity analysis | 1296 family–policy–setting records from one fixed cohort |
| Generated-caption application | 50 prespecified LLaVA-1.5 descriptions; cached generation/extraction reused, downstream decisions actually executed |
| Fresh timing | 10 AMBER-Attribute images plus one warm-up; native cached-prior path |

The supplementary empirical Full gate exceeds the 5% test IIR budget in one
of three families. LTT takes no action in all three. Neither generated-caption
policy accepts an edit, so unchanged output provides no evidence of edited-text
fluency or visual semantic correctness. These outcomes are retained in the
release.

A full cleaned-index MMRel intervention rerun, replication of the supplement
across the other backbones, larger intervention strata, and edited-output
image-semantic evaluation are not represented as completed experiments.
Archived original-index intervention outputs are not cleaned-index efficacy
evidence.

## Reanalyse the fixed supplement

This path uses released scores and final-layer features; it does not rerun
VLM inference. Install the main dependencies and the package first:

```text
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/18_unpack_supplement_snapshot.py --config configs/default.yaml
python analyse_supplement.py --config configs/default.yaml
```

The snapshot is unpacked into `work/supplement_case_study/`. Analysis outputs
are written beneath that workspace, separately from full-pipeline outputs.
The one-backbone protocol and its workspace overrides are defined inside
`configs/default.yaml`; there is no second experiment configuration file.
The original scientific analysis was executed during revision. The portable
upload wrappers have not been newly executed during packaging.

See [reproducibility details](docs/REPRODUCIBILITY.md) for the raw-input boundary
and the optional GPU entry points.

## Full pipeline and external resources

The main implementation supports LLaVA-1.5, LLaVA-1.6, InstructBLIP and
Qwen-VL-Chat. Frozen CLIP and Grounding DINO provide visual evidence. Model
identifiers, download/cache locations and dataset paths are in
`configs/default.yaml`. Paths in this release are relative to the repository.
Set local checkpoint paths there only when using your own local resources.

Install a PyTorch build appropriate for your hardware before installing the
remaining main dependencies. The recorded supplementary environment used
PyTorch **2.12.1+cu126** and Transformers **4.49.0**; this is an execution record,
not a claim that every environment has been validated. The lightweight
extraction-table commands above require only PyYAML.

The numbered pipeline covers data acquisition, generation/extraction, visual
signals, one-shot internal confidence, sampled uncertainty, diagnostics,
residuals, conformal screening, index construction and selective intervention.
Reviewer-specific analyses are in scripts 14–17 and `src/hmqr/review/`.

```text
python scripts/00_download_data.py --config configs/default.yaml
```

AMBER/R-Bench resources may require manual placement. Follow
[data layout and licensing](docs/DATA.md). Retrieval uses Visual Genome JSON
annotations; decoded-pixel overlap screening additionally requires the image
corpus. Model weights and benchmark assets remain subject to their providers'
licenses.

`run_all.py` is the optional full runner. A full run involving sampled entropy
and internal confidence requires explicit call budgets. Inspect pending counts
before supplying budgets:

```text
python scripts/02_semantic_entropy.py --config configs/default.yaml
python scripts/02_internal_confidence.py --config configs/default.yaml
```

These commands report pending workload unless their explicit apply option is
given. Avoid mixing full runs with the released supplementary workspace or
archived results.

## Interpretation

- Benchmark labels are the only source of native evaluation truth. Unresolved
  claims have label `-1` and are excluded from supervised metric denominators.
- Recall calibration uses a lower-tail conformal rank and the declared
  image-level sampling unit. It does not by itself certify intervention risk.
- Empirical gating and the separate LTT comparison have different assumptions
  and claims. Neither is an arbitrary-domain-shift guarantee.
- Native annotation-derived targets, textual extraction fidelity and the
  factual quality of generated/edited captions are different evaluations.
- Full and no-prior residual models are fitted separately. Prior availability
  is recorded independently from its numeric value.

## License

The implementation retains the repository's existing [MIT License](LICENSE),
including its original copyright notice. Third-party models and datasets retain
their own terms. See [data notes](docs/DATA.md) for the released derived records.
