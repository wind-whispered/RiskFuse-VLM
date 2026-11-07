"""Reproducible full-corpus pixel fingerprints for retrieval contamination audit."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn

from ..utils import read_jsonl, write_json

SCHEMA = 1

def fingerprint(path):
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert('RGB')
        exact = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        hashes = []
        for image in (rgb, ImageOps.mirror(rgb)):
            grey = np.asarray(image.convert('L').resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
            coefficients = dctn(grey, type=2, norm='ortho')[:8, :8].reshape(-1)
            bits = coefficients > np.median(coefficients[1:])
            value = 0
            for bit in bits: value = (value << 1) | int(bit)
            hashes.append(f'{value:016x}')
        return {'rgb_sha256': exact, 'phash': hashes[0], 'phash_mirror': hashes[1]}

def _one(task):
    identifier, path = task
    try:
        stat = Path(path).stat()
        return dict(vg_image_id=identifier, path=str(path), size_bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, schema_version=SCHEMA, **fingerprint(path))
    except Exception as exc:
        return {'vg_image_id':identifier, 'path':str(path), 'error':str(exc), 'schema_version':SCHEMA}

def build_corpus(cfg):
    settings = cfg['datasets']['visual_genome']
    roots = [cfg.abspath(p) for p in settings.get('pixel_audit_roots', [settings['images_root']])]
    metadata = cfg.abspath(cfg['datasets']['visual_genome']['root']) / 'image_data.json'
    meta = json.loads(metadata.read_text(encoding='utf-8'))
    expected = {str(int(r.get('image_id', r.get('id')))) for r in meta}
    out = cfg.path('review') / 'reviewer_revision'
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'vg_pixel_fingerprints.jsonl'
    records = {}
    if cache.exists():
        for row in read_jsonl(cache):
            if row.get('schema_version') == SCHEMA and not row.get('error'):
                path = Path(row['path'])
                if path.is_file():
                    stat = path.stat()
                    if stat.st_size == row['size_bytes'] and stat.st_mtime_ns == row['mtime_ns']:
                        records[row['vg_image_id']] = row
    paths = {}
    for root in roots:
        for path in root.rglob('*'):
            if path.suffix.lower() in ('.jpg','.jpeg','.png') and path.stem.isdigit():
                identifier = str(int(path.stem))
                if identifier in expected:
                    paths.setdefault(identifier,path)
    pending = [(i, paths[i]) for i in sorted(expected & paths.keys()) if i not in records]
    print(f'VG_PIXEL_AUDIT expected={len(expected)} available={len(paths)} cached={len(records)} pending={len(pending)}',flush=True)
    errors = []
    workers = int(cfg.get('reviewer_revision',{}).get('pixel_hash_workers',8))
    with cache.open('a',encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        for count, row in enumerate(pool.map(_one, pending),1):
            stream.write(json.dumps(row)+'\n')
            if row.get('error'): errors.append(row)
            else: records[row['vg_image_id']] = row
            if count % 2000 == 0:
                stream.flush()
                print(f'VG_PIXEL_AUDIT computed={count}/{len(pending)}',flush=True)
    report = {'schema_version':SCHEMA,'expected':len(expected),'fingerprinted':len(expected & records.keys()),
              'missing_vg_ids':sorted(expected - records.keys()),'errors':errors,
              'complete':expected <= records.keys(),'method':'RGB_SHA256_and_64bit_DCT_pHash_plus_horizontal_flip'}
    write_json(out/'vg_pixel_corpus_manifest.json',report)
    return list(records.values()),report

class PixelSearch:
    def __init__(self,records,radius):
        self.radius = int(radius)
        if self.radius != 4: raise ValueError('The fixed five-band search is specified for radius four')
        self.exact = {}
        self.bands = {}
        self.hashes = {}
        for row in records:
            key = row['vg_image_id']; value = int(row['phash'],16)
            self.exact.setdefault(row['rgb_sha256'],set()).add(key)
            self.hashes[key] = value
            for band,(offset,width) in enumerate(((0,13),(13,13),(26,13),(39,13),(52,12))):
                self.bands.setdefault((band,(value>>offset)&((1<<width)-1)),set()).add(key)

    def query(self,path):
        record = fingerprint(path)
        exact = set(self.exact.get(record['rgb_sha256'],set()))
        matches = set(exact); minimum = None
        for raw in (record['phash'],record['phash_mirror']):
            value = int(raw,16);candidates = set()
            for band,(offset,width) in enumerate(((0,13),(13,13),(26,13),(39,13),(52,12))):
                candidates.update(self.bands.get((band,(value>>offset)&((1<<width)-1)),set()))
            for identifier in candidates:
                distance = (value ^ self.hashes[identifier]).bit_count()
                if distance <= self.radius:
                    matches.add(identifier)
                    minimum = distance if minimum is None else min(minimum,distance)
        return dict(record, exact_vg_ids=sorted(exact), candidate_vg_ids=sorted(matches), minimum_matched_phash_distance=minimum)

def build_search(cfg):
    records, report = build_corpus(cfg)
    return PixelSearch(records,cfg.get('reviewer_revision',{}).get('pixel_phash_radius',4)),report
