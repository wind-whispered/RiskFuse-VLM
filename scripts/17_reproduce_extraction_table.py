#!/usr/bin/env python
"""Recompute the human-reviewed extraction table from released final judgements.

This is an actual CPU analysis entry point. It needs Python and PyYAML, not
models, images, torch, or cached summary metrics. All paths come from the unique
configuration. Source records and predictions are independently joined by task.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import yaml


def records(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/default.yaml')
    parser.add_argument('--output-dir',default=None)
    args=parser.parse_args()
    config_path=Path(args.config).resolve()
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    root=config_path.parent.parent
    settings=cfg['reviewer_revision']
    review_path=root/settings['assisted_review_records']
    source_path=root/settings['assisted_review_tasks']
    audit=records(review_path)
    raw_tasks=records(source_path)
    tasks={r['task_id']:r for r in raw_tasks}
    expected=int(settings['annotation_images_per_backbone'])*len(cfg['backbones']['order'])
    identifiers=[r['task_id'] for r in audit]
    if len(audit)!=expected or len(set(identifiers))!=expected or len(tasks)!=len(raw_tasks) or set(identifiers)!=set(tasks):
        raise ValueError('The released fixed audit must contain exactly one completed judgement per source task')
    counts=defaultdict(lambda:{'true_positive':0,'n_predicted':0,'n_reference':0})
    for item in audit:
        source=tasks[item['task_id']]
        if not item.get('reviewer') or item['task_sha256']!=source['task_sha256']:
            raise ValueError('Completed review provenance does not match the source')
        refs={r['candidate_id']:r['reference'] for r in item['actions'] if r['action']!='删除'}
        predictions={r['claim_id']:r for r in source['predictions']}
        matches={r['predicted_claim_id']:r['reference_id'] for r in item['matches']}
        if set(matches)!=set(predictions):raise ValueError('Missing final prediction match')
        used=set()
        local=defaultdict(lambda:{'true_positive':0,'n_predicted':0,'n_reference':0})
        for ref in refs.values():local[ref['claim_type']]['n_reference']+=1
        for identifier,prediction in predictions.items():
            kind=prediction['claim_type'];local[kind]['n_predicted']+=1
            matched=matches[identifier]
            if matched is None:continue
            if matched in used or matched not in refs or refs[matched]['claim_type']!=kind:
                raise ValueError('The final semantic match must be one-to-one and same-type')
            used.add(matched);local[kind]['true_positive']+=1
        for kind,values in local.items():
            for group in ((item['backbone'],kind),('ALL',kind),('ALL','ALL')):
                for key,value in values.items():counts[group][key]+=value
    for backbone in [*cfg['backbones']['order'],'ALL']:
        for kind in ('attr','rel'):counts[(backbone,kind)]
    rows=[]
    for (backbone,kind),values in sorted(counts.items()):
        tp=values['true_positive'];p=values['n_predicted'];n=values['n_reference']
        rows.append(dict(backbone=backbone,claim_type=kind,**values,
                         precision=tp/p if p else None,recall=tp/n if n else None,
                         f1=2*tp/(p+n) if p+n else None))
    out=Path(args.output_dir).resolve() if args.output_dir else root/'results/reproduced_extraction'
    out.mkdir(parents=True,exist_ok=True)
    csv_path=out/'revision_extraction_audit.csv'
    with csv_path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['backbone','claim_type','true_positive','n_predicted','n_reference','precision','recall','f1'])
        writer.writeheader();writer.writerows(rows)
    manifest={'n_completed_descriptions':len(audit),'n_shared_images':len({r['image_id'] for r in audit}),
              'review_records_sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),
              'source_tasks_sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
              'table_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
              'analysis':'Pooled counts recomputed from final reference actions and one-to-one matches; no stored summary counts used',
              'annotation_mode':'AI-assisted, human-reviewed; one reviewer; no independent or blinded annotation claim'}
    (out/'reproduction_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
