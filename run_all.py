#!/usr/bin/env python
"""Run the whole pipeline with one command -- pure Python, no shell required.

This is the portable alternative to run_all.sh / Makefile: it works the same on
Ubuntu, Windows and inside PyCharm ("Run" on this file), and needs no `bash`,
`sh` or `make`.

    python run_all.py --semantic-budget N --internal-budget N
                                            # steps 00..12, budget guarded
    python run_all.py --from 03             # resume at step 03
    python run_all.py --from 05             # no model-call budgets needed
    python run_all.py --from 03 --to 06     # run a range
    python run_all.py --only 12             # a single step
    python run_all.py --smoke               # tiny end-to-end run (see below)
    python run_all.py --skip-deps           # do not let step 00 pip install
    python run_all.py --dry-run             # just print the commands

Each step is a separate `python scripts/NN_*.py` process, so a crash in one step
never poisons the interpreter state of the next. Steps are idempotent: finished
work is detected and skipped. Full runs that include 02s/02i additionally require
explicit model-call budgets, so a resume cannot silently launch an expensive job.

--smoke caps the workload (subset_limit, --max-rows). The four external
datasets provide evaluation labels; Visual Genome remains the retrieval-only
knowledge base. The offline tests/smoke_test.py checks plumbing and rendering.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"

# (step id, filename, human label, extra args in smoke mode)
STEPS = [
    ("00", "00_download_data.py", "download datasets + models", []),
    ("01", "01_generate_and_extract.py", "generate captions, extract claims, label", []),
    ("02", "02_compute_signals.py", "grounding signals ALD / RLD / global / l_p", []),
    ("02s", "02_semantic_entropy.py", "budgeted matched SemanticEntropy baselines", []),
    ("02i", "02_internal_confidence.py", "budgeted one-shot internal confidence", []),
    ("03", "03_layer_probe.py", "layer-wise probes (Figure 1)", ["--max-rows", "120"]),
    ("04", "04_attention.py", "attention analysis (Figure 2)", ["--max-rows", "80"]),
    ("05", "05_residual.py", "residual decomposition (Section 3.4, Figure 3)", []),
    ("06", "06_conformal_detect.py", "conformal detection (RQ1, Figure 5)", []),
    ("07", "07_build_vg_index.py", "Visual Genome RAG index", []),
    ("08", "08_run_pipeline.py", "selective RAG revision pipeline", []),
    ("09", "09_rq2_gate.py", "RQ2 residual gate study (Figure 6)", []),
    ("10", "10_rq3_crossbackbone.py", "RQ3 cross-backbone + quality (Figure 7, Table 1)", []),
    ("11", "11_rq4_ablation.py", "RQ4 ablation (Figure 8, Table 2)", []),
    ("12", "12_make_all_figures.py", "render seven figures and two tables", []),
]


# 20, not 8: a discriminative dataset yields one claim per image, and step 08
# skips any claim type with fewer than MIN_CLAIMS_FOR_CP (10) rows because a
# conformal quantile over a handful of points is meaningless. At 8 images per
# dataset amber_attr / rbench / mmrel produced "0it" and the revision path for
# discriminative claims was never exercised by the smoke run at all.
SMOKE_IMAGES_PER_DATASET = 20


def make_smoke_config(base_config: str) -> str:
    """Write a reduced copy of the config next to the original and return its path.

    Steps 01/02 take their workload from `datasets.subset_limit` in the YAML (they
    have no command-line cap), so a genuine smoke run has to go through the config.
    """
    import yaml

    src = Path(base_config)
    if not src.is_absolute():
        src = ROOT / src
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("datasets", {})["subset_limit"] = SMOKE_IMAGES_PER_DATASET
    cfg.setdefault("generation", {})["max_new_tokens"] = 96

    # Send the smoke outputs to their own tree. Every step skips work whose
    # output file already exists, so smoke artefacts left in data/processed
    # would be picked up by the following full run and its numbers would come
    # from 8 images per dataset without a single warning.
    # data/datasets and data/indexes stay shared on purpose: both are built from the
    # benchmark releases alone and do not depend on subset_limit, so rebuilding
    # them for the smoke run would only waste time.
    paths = cfg.setdefault("paths", {})
    paths["processed"] = "data/processed_smoke"
    paths["results"] = "results_smoke"
    paths["figures"] = "results_smoke/figures"
    paths["tables"] = "results_smoke/tables"
    paths["logs"] = "results_smoke/logs"
    paths["review"] = "results_smoke/review"

    dst = src.parent / "_smoke_generated.yaml"
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[smoke] wrote {dst} (subset_limit={SMOKE_IMAGES_PER_DATASET} images/dataset)")
    return str(dst.relative_to(ROOT)) if dst.is_relative_to(ROOT) else str(dst)


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--from", dest="start", default="00", help="first step id, e.g. 03")
    p.add_argument("--to", dest="end", default="12", help="last step id, e.g. 06")
    p.add_argument("--only", default=None, help="run exactly one step id")
    p.add_argument("--overwrite", action="store_true", help="pass --overwrite to every step")
    p.add_argument("--skip-deps", action="store_true", help="step 00 will not pip install")
    p.add_argument("--semantic-budget", type=int, default=None,
                   help="maximum sampled sequences authorised for step 02s")
    p.add_argument("--internal-budget", type=int, default=None,
                   help="maximum unique VLM generations authorised for step 02i")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end validation run")
    p.add_argument(
        "--allow-subset", action="store_true",
        help="allow a non-null datasets.subset_limit outside --smoke (never use for paper results)",
    )
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument("--keep-going", action="store_true",
                   help="continue after a failing step instead of stopping")
    args = p.parse_args()

    if args.only:
        selected = [s for s in STEPS if s[0] == args.only]
        if not selected:
            print(f"unknown step id '{args.only}'; valid: {[s[0] for s in STEPS]}")
            return 2
    else:
        selected = [s for s in STEPS if args.start <= s[0] <= args.end]

    selected_ids = {step[0] for step in selected}
    missing_budgets = []
    if not args.smoke and not args.dry_run:
        if "02s" in selected_ids and args.semantic_budget is None:
            missing_budgets.append("--semantic-budget")
        if "02i" in selected_ids and args.internal_budget is None:
            missing_budgets.append("--internal-budget")
    if missing_budgets:
        print(
            "Refusing to start before explicit generation budgets are supplied: "
            + ", ".join(missing_budgets)
        )
        print("Estimate without loading models:")
        print("  python scripts/02_semantic_entropy.py --config " + args.config)
        print("  python scripts/02_internal_confidence.py --config " + args.config)
        return 2

    config = args.config
    if args.smoke and not args.dry_run:
        config = make_smoke_config(args.config)

    if not args.smoke:
        import yaml

        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = ROOT / config_path
        with open(config_path, "r", encoding="utf-8") as stream:
            configured = yaml.safe_load(stream)
        subset_limit = configured.get("datasets", {}).get("subset_limit")
        if subset_limit is not None and not args.allow_subset:
            print(
                "Refusing a nominal full run with datasets.subset_limit="
                f"{subset_limit}. Set it to null for paper results, use --smoke, "
                "or explicitly pass --allow-subset for a non-paper diagnostic run."
            )
            return 2

    print("=" * 70)
    print("hmqr pipeline")
    print(f"  python : {sys.executable}")
    print(f"  root   : {ROOT}")
    print(f"  config : {config}")
    print(f"  steps  : {', '.join(s[0] for s in selected)}")
    if args.smoke:
        print("  mode   : SMOKE (reduced workload)")
    else:
        print("  mode   : FULL (datasets.subset_limit=null)")
    print("=" * 70)

    failed: list[str] = []
    t_all = time.time()

    for step_id, filename, label, smoke_args in selected:
        script = SCRIPTS / filename
        if not script.exists():
            print(f"\n[{step_id}] MISSING SCRIPT {script}")
            failed.append(step_id)
            if not args.keep_going:
                return 1
            continue

        cmd = [sys.executable, str(script), "--config", config]
        # The VG index is intentionally shared by smoke and full runs because
        # it is independent of subset_limit.  Never let a smoke overwrite
        # rebuild (or later replace with a reduced index) this shared asset.
        protect_shared_vg_index = args.smoke and step_id == "07"
        if args.overwrite and not protect_shared_vg_index:
            cmd.append("--overwrite")
        if step_id == "00" and args.skip_deps:
            cmd.append("--no-deps")
        if step_id == "02s":
            budget = (
                args.semantic_budget if args.semantic_budget is not None
                else ("<REQUIRED>" if args.dry_run else 100000)
            )
            cmd += ["--apply", "--max-generation-calls", str(budget)]
        if step_id == "02i":
            budget = (
                args.internal_budget if args.internal_budget is not None
                else ("<REQUIRED>" if args.dry_run else 100000)
            )
            cmd += ["--apply", "--max-generation-calls", str(budget)]
        if args.smoke:
            cmd += smoke_args

        print(f"\n{'-' * 70}\n[{step_id}] {label}\n      {' '.join(cmd)}\n{'-' * 70}")
        if args.overwrite and protect_shared_vg_index:
            print("[smoke] preserving shared data/indexes/visual_genome; --overwrite was not passed to step 07")
        if args.dry_run:
            continue

        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(ROOT))
        dt = time.time() - t0
        if rc == 0:
            print(f"[{step_id}] done in {_fmt(dt)}")
        else:
            print(f"[{step_id}] FAILED (exit {rc}) after {_fmt(dt)}")
            failed.append(step_id)
            if not args.keep_going:
                print(f"\nStopped at step {step_id}. Fix the error and resume with:\n"
                      f"    python run_all.py --from {step_id}")
                return rc

    print("\n" + "=" * 70)
    print(f"total wall clock: {_fmt(time.time() - t_all)}")
    if failed:
        print(f"FAILED STEPS: {', '.join(failed)}")
        return 1
    if args.dry_run:
        print("DRY RUN COMPLETE -- no scripts were executed and no artifacts were produced")
        print("=" * 70)
        return 0
    print("ALL STEPS COMPLETED")
    print("  figures -> results/figures/figure{1,2,3,5,6,7,8}.pdf + .png (600 dpi)")
    print("  tables  -> results/tables/table{1,2}.pdf + .tex + .csv")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
