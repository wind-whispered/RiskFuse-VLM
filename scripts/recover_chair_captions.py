#!/usr/bin/env python
"""Recover CHAIR captions for images that produced zero atomic claims."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hmqr.backbones.registry import build_backbone  # noqa: E402
from hmqr.config import load_config, seed_everything  # noqa: E402
from hmqr.data.datasets import load_dataset  # noqa: E402
from hmqr.data.images import load_image  # noqa: E402
from hmqr.utils import read_jsonl, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--backbone", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.backbone not in cfg.backbone_names:
        parser.error(f"unknown backbone: {args.backbone}")
    seed_everything(cfg.get("seed", 42))

    processed = cfg.path("processed")
    claims = read_jsonl(processed / f"claims_{args.backbone}_chair500.jsonl")
    supplemental_path = processed / f"chair_captions_{args.backbone}.jsonl"
    supplemental = read_jsonl(supplemental_path) if supplemental_path.exists() else []
    available = {
        str(row["image_id"])
        for row in [*claims, *supplemental]
        if row.get("caption")
    }
    samples = [
        sample for sample in load_dataset(cfg, "chair500")
        if str(sample.image_id) not in available
    ]
    print(f"{args.backbone}: missing_zero_claim_captions={len(samples)}")
    if not samples:
        return 0

    backbone = build_backbone(cfg, args.backbone, load=True)
    prompt = cfg["generation"]["caption_prompt"]
    max_new_tokens = int(cfg["generation"]["max_new_tokens"])
    recovered = list(supplemental)
    for sample in tqdm(samples, desc=f"recover {args.backbone} CHAIR", ncols=88):
        image = load_image(sample.image_path)
        if image is None:
            raise RuntimeError(f"cannot load CHAIR image {sample.image_id}")
        result = backbone.generate(image, prompt, max_new_tokens=max_new_tokens)
        caption = str(result.get("text", "")).strip()
        if not caption:
            raise RuntimeError(f"empty recovered caption for CHAIR image {sample.image_id}")
        recovered.append({
            "image_id": sample.image_id,
            "caption": caption,
            "source": "targeted_stage01_zero_claim_recovery",
            "schema_version": 1,
        })
        write_jsonl(supplemental_path, recovered)
    backbone.unload()
    print(f"wrote {supplemental_path} ({len(recovered)} supplemental captions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
