#!/usr/bin/env python
"""Read-only final audit for the seven figures and two manuscript tables."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmqr.config import load_config  # noqa: E402
from hmqr.figures.style import OUTPUT_DPI, figure_size  # noqa: E402

FIGURES = ("figure1", "figure2", "figure3", "figure5", "figure6", "figure7", "figure8")
TABLES = ("table1", "table2")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_step12():
    spec = importlib.util.spec_from_file_location(
        "hmqr_step12_delivery", ROOT / "scripts" / "12_make_all_figures.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _check_pdf_fonts(path: Path, errors: list[str], warnings: list[str]) -> None:
    pdffonts = shutil.which("pdffonts")
    if pdffonts:
        result = subprocess.run(
            [pdffonts, str(path)], capture_output=True, text=True, check=False
        )
        if result.returncode:
            errors.append(f"{path.name}: pdffonts failed: {result.stderr.strip()}")
        elif "Type 3" in result.stdout:
            errors.append(f"{path.name}: contains a prohibited Type-3 font")
        return
    if b"/Subtype /Type3" in path.read_bytes():
        errors.append(f"{path.name}: contains a prohibited Type-3 font")
    else:
        warnings.append(
            f"{path.name}: pdffonts unavailable; only a raw Type-3 signature scan was possible"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    step12 = _load_step12()
    errors: list[str] = []
    warnings: list[str] = []

    try:
        step12._validate_plot_inputs(cfg, step12.PLOT_INPUTS)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"input contract failed: {exc}")

    required: list[Path] = [cfg.path("tables") / name for name in step12.PLOT_INPUTS]
    for name in FIGURES:
        pdf = cfg.path("figures") / f"{name}.pdf"
        png = cfg.path("figures") / f"{name}.png"
        required.extend([pdf, png])
        if not pdf.exists() or pdf.stat().st_size == 0:
            errors.append(f"missing/empty {pdf.relative_to(cfg.root)}")
        else:
            if not pdf.read_bytes().startswith(b"%PDF"):
                errors.append(f"{pdf.name}: invalid PDF header")
            _check_pdf_fonts(pdf, errors, warnings)
        if not png.exists() or png.stat().st_size == 0:
            errors.append(f"missing/empty {png.relative_to(cfg.root)}")
        else:
            try:
                with Image.open(png) as image:
                    dpi = tuple(float(value) for value in image.info.get("dpi", (0.0, 0.0)))
                    pixel_size = tuple(int(value) for value in image.size)
                expected = tuple(
                    int(round(value * OUTPUT_DPI)) for value in figure_size(name)
                )
                if len(dpi) != 2 or any(abs(value - OUTPUT_DPI) > 1.0 for value in dpi):
                    errors.append(
                        f"{png.name}: dpi={dpi}, expected {OUTPUT_DPI}x{OUTPUT_DPI}"
                    )
                if any(abs(actual - target) > 1 for actual, target in zip(pixel_size, expected)):
                    errors.append(
                        f"{png.name}: pixels={pixel_size}, expected approximately {expected}"
                    )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{png.name}: unreadable PNG: {exc}")

    for name in TABLES:
        for suffix in ("pdf", "tex", "csv"):
            path = cfg.path("tables") / f"{name}.{suffix}"
            required.append(path)
            if not path.exists() or path.stat().st_size == 0:
                errors.append(f"missing/empty {path.relative_to(cfg.root)}")
            elif suffix == "pdf":
                if not path.read_bytes().startswith(b"%PDF"):
                    errors.append(f"{path.name}: invalid PDF header")
                _check_pdf_fonts(path, errors, warnings)
            elif suffix == "tex":
                source = path.read_text(encoding="utf-8")
                tabular_lines = [
                    line for line in source.splitlines() if r"\begin{tabular}" in line
                ]
                if not tabular_lines or r"\end{tabular}" not in source:
                    errors.append(f"{path.name}: missing editable tabular environment")
                if r"\resizebox" in source:
                    errors.append(f"{path.name}: table must not use whole-table resizebox scaling")
                if any("|" in line for line in tabular_lines):
                    errors.append(f"{path.name}: vertical table rules are not permitted")
                if r"\scriptsize" in source or r"\tiny" in source:
                    errors.append(f"{path.name}: table text is smaller than normal manuscript size")

    table1_csv = cfg.path("tables") / "table1.csv"
    table2_csv = cfg.path("tables") / "table2.csv"
    if table1_csv.exists():
        table1 = pd.read_csv(table1_csv)
        expected_pairs = {
            (backbone, dataset)
            for backbone in {
                "LLaVA-1.5-7B", "LLaVA-1.6-7B", "InstructBLIP-7B", "Qwen-VL-Chat"
            }
            for dataset in {
                "AMBER-Attribute", "AMBER-Relation", "R-Bench", "MMRel", "CHAIR-500"
            }
        }
        actual_pairs = set(table1[["backbone", "dataset"]].itertuples(index=False, name=None))
        if actual_pairs != expected_pairs:
            errors.append("table1.csv must contain every 4-backbone x 5-dataset pair exactly once")
    if table2_csv.exists():
        table2 = pd.read_csv(table2_csv)
        expected_variants = {
            "Full",
            "No residual gate",
            "Fixed threshold (no CP)",
            "No residualisation",
            "No VG prior",
            "Abstain all",
        }
        expected_columns = {
            "variant", "cp", "residual", "kb", "gate", "intervene",
            "target_hallucination_removal_rate",
            "incorrect_intervention_rate",
            "net_hallucination_reduction",
            "output_coverage",
        }
        if len(table2) != len(expected_variants) or set(table2.get("variant", [])) != expected_variants:
            errors.append("table2.csv must contain the six specified intervention variants")
        missing_columns = expected_columns - set(table2.columns)
        if missing_columns:
            errors.append(
                "table2.csv is missing checkbox/metric columns: "
                + ", ".join(sorted(missing_columns))
            )

    overhead = cfg.path("logs") / "overhead.txt"
    required.append(overhead)
    if not overhead.exists() or overhead.stat().st_size == 0:
        errors.append("missing/empty results/logs/overhead.txt")
    elif (
        "# computational overhead (Section 5.5)" not in overhead.read_text(encoding="utf-8")
        or f"pipeline schema version   : {step12.PIPELINE_SCHEMA_VERSION}"
        not in overhead.read_text(encoding="utf-8")
    ):
        errors.append("overhead.txt is stale or does not identify manuscript Section 5.5")

    manifest_path = cfg.path("logs") / "artifact_manifest.json"
    if not manifest_path.exists():
        errors.append("missing results/logs/artifact_manifest.json")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            config_path = Path(args.config)
            if not config_path.is_absolute():
                config_path = cfg.root / config_path
            if manifest.get("config_sha256") != _sha256(config_path):
                errors.append("artifact manifest config hash does not match current config")
            indexed = {str(item["path"]): item for item in manifest.get("files", [])}
            for path in required:
                if not path.exists():
                    continue
                key = path.relative_to(cfg.root).as_posix()
                item = indexed.get(key)
                if item is None:
                    errors.append(f"manifest missing {key}")
                elif item.get("sha256") != _sha256(path):
                    errors.append(f"manifest hash mismatch for {key}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"invalid artifact manifest: {exc}")

    print("\n================ final delivery audit ================")
    print(f"figures: {len(FIGURES)} PDF + {len(FIGURES)} PNG ({OUTPUT_DPI} dpi)")
    print(f"tables:  {len(TABLES)} PDF + TeX + CSV")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        print(f"RESULT: FAILED ({len(errors)} issue(s))")
        return 1
    print("RESULT: PASS -- final seven-figure/two-table package is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
