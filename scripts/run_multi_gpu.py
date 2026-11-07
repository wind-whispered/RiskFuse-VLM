#!/usr/bin/env python
"""Run paper stages on four independent GPUs, one frozen VLM backbone per GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))

from hmqr.config import load_config  # noqa: E402
BACKBONES = ("llava15", "llava16", "instructblip", "qwenvl")
GPU_MAP = dict(zip(BACKBONES, (0, 1, 2, 3)))
SHARDED = {
    "01": "01_generate_and_extract.py",
    "02": "02_compute_signals.py",
    "03": "03_layer_probe.py",
    "04": "04_attention.py",
    "02s": "02_semantic_entropy.py",
    "02i": "02_internal_confidence.py",
    "08": "08_run_pipeline.py",
}
MASTER = {
    "05": "05_residual.py",
    "06": "06_conformal_detect.py",
    "07": "07_build_vg_index.py",
    "09": "09_rq2_gate.py",
    "10": "10_rq3_crossbackbone.py",
    "11": "11_rq4_ablation.py",
}
PHASES = ("01", "02", "03", "04", "02s", "02i", "05", "06", "07", "08", "09", "10", "11")


def _write_status(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _budgeted_pending(
    phase: str,
    config: str,
    overwrite: bool,
    revision_dataset: str | None = None,
) -> dict[str, int]:
    cfg = load_config(config)
    if phase == "02s":
        from hmqr.signals.semantic_runner import estimate_calls

        n_samples = int(cfg.get("semantic_entropy", {}).get("n_samples", 10))
        return {
            backbone: sum(estimate_calls(
                cfg, [backbone], cfg["datasets"]["detect"], n_samples, overwrite
            ).values())
            for backbone in BACKBONES
        }
    if phase == "02i":
        from hmqr.signals.lp_runner import estimate_calls

        datasets = list(cfg["datasets"]["detect"]) + list(
            cfg.get("datasets", {}).get("quality", [])
        )
        return {
            backbone: sum(estimate_calls(cfg, [backbone], datasets, overwrite).values())
            for backbone in BACKBONES
        }
    if phase == "08":
        from hmqr.revise.pipeline import preflight_one

        datasets = (
            [revision_dataset]
            if revision_dataset
            else ["amber_attr", "amber_rel", "rbench", "mmrel", "chair500"]
        )
        pending: dict[str, int] = {}
        for backbone in BACKBONES:
            total = 0
            for dataset in datasets:
                report = preflight_one(
                    cfg, backbone, dataset, strategy="full",
                    overwrite=overwrite,
                )
                value = report.get("pending_next_run_generations")
                if value is None:
                    raise RuntimeError(
                        f"phase 08 budget is unresolved for {backbone}/{dataset}: "
                        f"{report.get('status')} {report.get('reason', '')}"
                    )
                total += int(value)
            pending[backbone] = total
        return pending
    return {}


def run_sharded(
    phase: str,
    config: str,
    overwrite: bool,
    stagger: int,
    online: bool,
    generation_budget: int | None = None,
    revision_dataset: str | None = None,
) -> bool:
    log_dir = load_config(config).path("logs") / "multigpu"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    jobs = []
    pending_by_backbone = _budgeted_pending(
        phase, config, overwrite, revision_dataset=revision_dataset
    )
    if pending_by_backbone:
        total_pending = sum(pending_by_backbone.values())
        if generation_budget is None:
            raise RuntimeError(f"phase {phase} requires an explicit generation budget")
        if total_pending > int(generation_budget):
            raise RuntimeError(
                f"phase {phase} budget exceeded: pending={total_pending} > "
                f"authorised={generation_budget}; by_backbone={pending_by_backbone}"
            )
        print(
            f"phase {phase}: total_pending={total_pending}, "
            f"authorised={generation_budget}, by_backbone={pending_by_backbone}"
        )
    for i, backbone in enumerate(BACKBONES):
        gpu = GPU_MAP[backbone]
        cmd = [
            sys.executable, str(SCRIPTS / SHARDED[phase]),
            "--config", config, "--backbone", backbone,
        ]
        if overwrite:
            cmd.append("--overwrite")
        if phase == "08" and revision_dataset:
            cmd += ["--dataset", revision_dataset]
        if phase in {"02s", "02i"}:
            pending = int(pending_by_backbone[backbone])
            if pending == 0:
                print(f"phase {phase}: {backbone} has no pending generations; skipped")
                continue
            cmd += ["--apply", "--max-generation-calls", str(pending)]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if not online:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        log_path = log_dir / f"phase{phase}_{backbone}.log"
        stream = open(log_path, "a", encoding="utf-8", buffering=1)
        stream.write(f"\n===== start {time.strftime('%Y-%m-%d %H:%M:%S')} cmd={cmd!r} =====\n")
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        jobs.append({
            "backbone": backbone, "gpu": gpu, "pid": proc.pid, "process": proc,
            "stream": stream, "log": str(log_path), "returncode": None,
        })
        print(f"phase {phase}: GPU {gpu} -> {backbone}, pid={proc.pid}, log={log_path}")
        if stagger and i + 1 < len(BACKBONES):
            time.sleep(stagger)

    while any(j["returncode"] is None for j in jobs):
        for job in jobs:
            if job["returncode"] is None:
                job["returncode"] = job["process"].poll()
        _write_status(
            status_path,
            {
                "phase": phase,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "jobs": [
                    {k: j[k] for k in ("backbone", "gpu", "pid", "log", "returncode")}
                    for j in jobs
                ],
            },
        )
        if any(j["returncode"] is None for j in jobs):
            time.sleep(5)

    for job in jobs:
        job["stream"].close()
    accepted = {0}
    failures = [j for j in jobs if j["returncode"] not in accepted]
    if failures:
        detail = ", ".join(f"{j['backbone']}={j['returncode']}" for j in failures)
        raise RuntimeError(f"phase {phase} shard failure(s): {detail}")

    if phase in {"02s", "02i"}:
        audit_script = (
            "02_semantic_entropy.py" if phase == "02s" else "02_internal_confidence.py"
        )
        subprocess.run(
            [
                sys.executable, str(SCRIPTS / audit_script), "--config", config,
                "--audit-existing", "--stamp-schema",
            ],
            cwd=ROOT, check=True,
        )
        return False

    merge_phase = "04" if phase in {"03", "04"} else phase
    if phase != "03":
        subprocess.run(
            [sys.executable, str(SCRIPTS / "merge_multi_gpu.py"),
             "--config", config, "--phase", merge_phase],
            cwd=ROOT, check=True,
        )
    return False


def run_master(phase: str, config: str, overwrite: bool, online: bool) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if not online:
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    cmd = [sys.executable, str(SCRIPTS / MASTER[phase]), "--config", config]
    if overwrite:
        cmd.append("--overwrite")
    print(f"master phase {phase}: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def run_revision_prep(
    kind: str,
    config: str,
    online: bool,
    stagger: int,
    generation_budget: int | None,
) -> bool:
    """Precompute expensive RQ revision variants in parallel before aggregation."""
    specs, phase = _revision_prep_specs(kind)
    pending_by_job = _revision_prep_pending(kind, config)
    total_pending = sum(pending_by_job.values())
    if generation_budget is None:
        raise RuntimeError(f"{phase} requires an explicit revision budget")
    if total_pending > int(generation_budget):
        raise RuntimeError(
            f"{phase} budget exceeded: pending={total_pending} > "
            f"authorised={generation_budget}; by_job={pending_by_job}"
        )
    print(
        f"{phase}: total_pending={total_pending}, authorised={generation_budget}, "
        f"by_job={pending_by_job}"
    )

    log_dir = load_config(config).path("logs") / "multigpu"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "status.json"
    jobs = []
    for i, (gpu, backbone, strategy, task_datasets) in enumerate(specs):
        cmd = [
            sys.executable, str(SCRIPTS / "run_revision_shard.py"),
            "--config", config, "--backbone", backbone, "--strategy", strategy,
        ]
        for dataset in task_datasets:
            cmd.extend(["--dataset", dataset])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if not online:
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
        log_path = log_dir / f"phase{phase}_gpu{gpu}_{strategy}.log"
        stream = open(log_path, "a", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
        )
        jobs.append({
            "backbone": f"{backbone}:{strategy}", "gpu": gpu, "pid": proc.pid,
            "process": proc, "stream": stream, "log": str(log_path), "returncode": None,
        })
        print(f"{phase}: GPU {gpu} -> {backbone}/{strategy}, pid={proc.pid}")
        if stagger and i + 1 < len(specs):
            time.sleep(stagger)

    while any(j["returncode"] is None for j in jobs):
        for job in jobs:
            if job["returncode"] is None:
                job["returncode"] = job["process"].poll()
        _write_status(
            status_path,
            {
                "phase": phase,
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "jobs": [
                    {k: j[k] for k in ("backbone", "gpu", "pid", "log", "returncode")}
                    for j in jobs
                ],
            },
        )
        if any(j["returncode"] is None for j in jobs):
            time.sleep(5)
    for job in jobs:
        job["stream"].close()
    failures = [j for j in jobs if j["returncode"] != 0]
    if failures:
        raise RuntimeError(
            f"{phase} failures: "
            + ", ".join(f"{j['backbone']}={j['returncode']}" for j in failures)
        )
    return False


def _revision_prep_specs(kind: str):
    if kind == "rq2":
        return ([
            (0, "llava15", "no_gate", ["amber_attr", "amber_rel"]),
            (2, "instructblip", "no_gate", ["amber_attr", "amber_rel"]),
        ], "09-prep")
    if kind == "rq4":
        return ([
            (1, "llava15", "abstain_all", ["amber_attr", "amber_rel"]),
            (2, "llava15", "no_rag", ["amber_attr", "amber_rel"]),
            (3, "llava15", "no_image", ["amber_attr", "amber_rel"]),
        ], "11-prep")
    raise ValueError(f"unknown revision-prep kind: {kind}")


def _revision_prep_pending(kind: str, config: str) -> dict[str, int]:
    """Return an exact model-free generation count for one RQ prep phase."""
    from hmqr.revise.pipeline import preflight_one

    specs, _ = _revision_prep_specs(kind)
    cfg = load_config(config)
    pending_by_job: dict[str, int] = {}
    for _, backbone, strategy, task_datasets in specs:
        key = f"{backbone}:{strategy}"
        total = 0
        for dataset in task_datasets:
            report = preflight_one(cfg, backbone, dataset, strategy=strategy)
            value = report.get("pending_next_run_generations")
            if value is None:
                _, phase = _revision_prep_specs(kind)
                raise RuntimeError(
                    f"{phase} budget is unresolved for "
                    f"{backbone}/{dataset}/{strategy}: "
                    f"{report.get('status')} {report.get('reason', '')}"
                )
            total += int(value)
        pending_by_job[key] = total
    return pending_by_job


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--phase", default="all",
        choices=PHASES + ("all",),
    )
    parser.add_argument(
        "--from", dest="from_phase", choices=PHASES,
        help="with --phase all, resume the complete workflow at this phase",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--semantic-budget", type=int, default=None)
    parser.add_argument("--internal-budget", type=int, default=None)
    parser.add_argument("--revision-budget", type=int, default=None)
    parser.add_argument(
        "--preflight", action="store_true",
        help="print exact model-free pending generation counts and exit",
    )
    parser.add_argument(
        "--revision-dataset",
        choices=("amber_attr", "amber_rel", "rbench", "mmrel", "chair500"),
        default=None,
        help="with --phase 08, restrict the run to one dataset family",
    )
    parser.add_argument("--stagger", type=int, default=20,
                        help="seconds between GPU process starts (default: 20)")
    parser.add_argument("--online", action="store_true",
                        help="allow Hugging Face network checks instead of cached offline mode")
    parser.add_argument(
        "--detach", action="store_true",
        help="launch this multi-GPU command in a new session and return immediately",
    )
    parser.add_argument(
        "--launcher-log", default=None,
        help="combined launcher log used with --detach",
    )
    args = parser.parse_args()
    if args.preflight and args.detach:
        parser.error("--preflight cannot be combined with --detach")
    if args.launcher_log and not args.detach:
        parser.error("--launcher-log requires --detach")
    if args.detach:
        forwarded = []
        skip_value = False
        for value in sys.argv[1:]:
            if skip_value:
                skip_value = False
                continue
            if value == "--detach":
                continue
            if value == "--launcher-log":
                skip_value = True
                continue
            forwarded.append(value)
        cfg = load_config(args.config)
        log_path = (
            Path(args.launcher_log)
            if args.launcher_log
            else cfg.path("logs") / "multigpu" / f"launcher_phase{args.phase}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", buffering=1) as stream:
            stream.write(
                f"\n===== detached start {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"args={forwarded!r} =====\n"
            )
            proc = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *forwarded],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"detached launcher pid={proc.pid} log={log_path}")
        return 0
    if args.from_phase and args.phase != "all":
        parser.error("--from cannot be combined with a specific --phase")
    if args.revision_dataset and args.phase != "08":
        parser.error("--revision-dataset requires --phase 08")
    if args.preflight:
        if args.phase == "08":
            pending = _budgeted_pending(
                "08", args.config, args.overwrite,
                revision_dataset=args.revision_dataset,
            )
            label = "phase08"
        elif args.phase == "09":
            pending = _revision_prep_pending("rq2", args.config)
            label = "phase09-prep"
        elif args.phase == "11":
            pending = _revision_prep_pending("rq4", args.config)
            label = "phase11-prep"
        else:
            parser.error("--preflight currently supports only phases 08, 09, and 11")
        print(
            f"{label}: total_pending={sum(pending.values())}, "
            f"by_job={pending}"
        )
        return 0

    selected_for_budget = (
        set(PHASES[PHASES.index(args.from_phase or "01"):])
        if args.phase == "all" else {args.phase}
    )
    if "02s" in selected_for_budget and args.semantic_budget is None:
        parser.error("phase 02s requires --semantic-budget")
    if "02i" in selected_for_budget and args.internal_budget is None:
        parser.error("phase 02i requires --internal-budget")
    if selected_for_budget.intersection({"08", "09", "11"}) and args.revision_budget is None:
        parser.error("phases 08, 09, and 11 require --revision-budget")

    if args.phase == "all":
        start = PHASES.index(args.from_phase or "01")
        selected = set(PHASES[start:])
        for phase in ("01", "02", "03", "04", "02s", "02i"):
            if phase in selected:
                budget = (
                    args.semantic_budget if phase == "02s"
                    else args.internal_budget if phase == "02i"
                    else None
                )
                run_sharded(
                    phase, args.config, args.overwrite, args.stagger, args.online, budget
                )
        for phase in ("05", "06", "07"):
            if phase not in selected:
                continue
            # Claims/signals were just rebuilt and merged. Residual/detection
            # tables from an older partial run must never be reused; the VG
            # index is dataset-only and remains safely reusable.
            force_analysis = phase in {"05", "06"}
            run_master(phase, args.config, args.overwrite or force_analysis, args.online)
        if "08" in selected:
            if run_sharded(
                "08", args.config, args.overwrite, args.stagger, args.online,
                args.revision_budget,
            ):
                return 2
        if "09" in selected:
            if run_revision_prep(
                "rq2", args.config, args.online, args.stagger,
                args.revision_budget,
            ):
                return 2
            run_master("09", args.config, True, args.online)
        if "10" in selected:
            run_master("10", args.config, True, args.online)
        if "11" in selected:
            if run_revision_prep(
                "rq4", args.config, args.online, args.stagger,
                args.revision_budget,
            ):
                return 2
            run_master("11", args.config, True, args.online)
    elif args.phase in SHARDED:
        budget = (
            args.semantic_budget if args.phase == "02s"
            else args.internal_budget if args.phase == "02i"
            else args.revision_budget if args.phase == "08"
            else None
        )
        if run_sharded(
            args.phase, args.config, args.overwrite, args.stagger, args.online,
            budget, revision_dataset=args.revision_dataset,
        ):
            return 2
    elif args.phase == "09":
        if run_revision_prep(
            "rq2", args.config, args.online, args.stagger,
            args.revision_budget,
        ):
            return 2
        run_master("09", args.config, True, args.online)
    elif args.phase == "11":
        if run_revision_prep(
            "rq4", args.config, args.online, args.stagger,
            args.revision_budget,
        ):
            return 2
        run_master("11", args.config, True, args.online)
    else:
        run_master(args.phase, args.config, args.overwrite, args.online)
    print("requested multi-GPU run completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
