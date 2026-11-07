"""Merge real GPU outputs and analyse the prespecified supplementary cohort."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,sys
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from supplement_common import load_supplement_config
from hmqr.config import load_config
from hmqr.utils import read_jsonl,write_jsonl,write_json
from hmqr.signals.compute import normalise_g
from hmqr.signals.lp_runner import mark_internal_confidence_selection
from hmqr.residual.decomposition import assign_claim_residuals
from hmqr.review.studies import groups_for_study,evaluate_group
from hmqr.review.audit import write_csv
from hmqr.signals.support import matched_signal_group

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='configs/default.yaml')
args=parser.parse_args()
cfg,ROOT=load_supplement_config(args.config);out=ROOT/'supplement_outputs'
for i in range(3):
    if not (out/f'worker_{i}_complete.json').is_file():raise RuntimeError('An actual cohort worker remains incomplete')
features={}
for i in range(3):features.update(json.loads((out/f'features_shard_{i}.json').read_text()))
if (out/'features_repaired.json').exists():features.update(json.loads((out/'features_repaired.json').read_text()))
for dataset in ('amber_attr','amber_rel'):
    source=read_jsonl(ROOT/'data/supplement'/f'claims_llava15_{dataset}.jsonl')
    joined={r['claim_id']:r for i in range(3) for r in read_jsonl(out/f'shard_{i}_{dataset}.jsonl')}
    if set(joined)!={r['claim_id'] for r in source}:raise RuntimeError('Incomplete actual inference cohort')
    for original in source:
        computed=joined[original['claim_id']]
        if any(original.get(k)!=computed.get(k) for k in ('claim_type','object','attribute','subject','relation','kb_prior','kb_prior_available','kb_prior_index_signature')):
            raise ValueError('Transported prior query or value changed during inference')
    rows=[joined[r['claim_id']] for r in source]
    normalise_g(rows);mark_internal_confidence_selection(rows,cfg)
    write_jsonl(cfg.path('processed')/f'claims_llava15_{dataset}.jsonl',rows)
assign_claim_residuals(cfg,prior_cache_signature=hashlib.sha256((cfg.path('vg_index')/'index_manifest.json').read_bytes()).hexdigest())
metrics=[];decisions=[];groups=[];coefs=[];probe=[]
for dataset in ('amber_attr','amber_rel'):
    path=cfg.path('processed')/f'claims_llava15_{dataset}.jsonl';rows=read_jsonl(path)
    by_group=defaultdict(list)
    for row in rows:by_group[matched_signal_group(row)].append(row)
    for group,items in by_group.items():
        usable=[r for r in items if features.get(r['claim_id'],{}).get('file')]
        train=[r for r in usable if r['split']=='calib_cp' and r['label'] in (0,1)]
        if len(train)>=10 and len({r['label'] for r in train})==2:
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))
            vector=lambda r:np.load(out/features[r['claim_id']]['file']).astype(np.float32)
            model.fit(np.stack([vector(r) for r in train]),[r['label'] for r in train])
            scores=model.predict_proba(np.stack([vector(r) for r in usable]))[:,1]
            for row,value in zip(usable,scores):row['probe_score']=float(value)
            probe.append(dict(dataset=dataset,group=group,recipe='fixed final layer; predicate if aligned, otherwise aligned entities',n_train=len(train),n_scored=len(usable)))
        representative=next((r for r in items if r.get('residual_intercept') is not None),None)
        if representative:
            coefs.append(dict(dataset=dataset,group=group,**{k:representative.get(k) for k in ('residual_intercept','residual_beta','residual_beta_kb','residual_beta_kb_available','residual_n_calibration','residual_design_rank')}))
    write_jsonl(path,rows)
    for group,parts in groups_for_study(rows,cfg).items():
        result,decision,status=evaluate_group(parts,cfg,dict(backbone='llava15',dataset=dataset,group=group))
        metrics.extend(result);decisions.extend(decision);groups.append(status)
write_csv(out/'supplement_matched_metrics.csv',metrics)
write_csv(out/'supplement_primary_decisions.csv',decisions)
write_csv(out/'supplement_residual_coefficients.csv',coefs)
primary=[r for r in metrics if r['alpha']==.1 and r['delta']==.05 and r['calibration_fraction']==1]
write_csv(out/'supplement_primary_metrics.csv',primary)
summary={'status':'complete' if metrics else 'complete_with_unsupported_groups',
         'scope':'LLaVA-1.5 supplementary AMBER case study; historical four-backbone experiments retained separately',
         'n_prespecified_rows':900,'n_metrics':len(metrics),'n_primary_cells':len(primary),'groups':groups,'probe':probe,
         'full_iir_exceedances':sum(r['test_iir_exceeds_budget'] for r in primary if r['method']=='Full'),
         'full_supported_cells':sum(r['method']=='Full' for r in primary),
         'ltt_no_action_cells':sum(r.get('gate_status')=='no_action' for r in primary if r['method']=='LTT residual gate'),
         'ltt_supported_cells':sum(r['method']=='LTT residual gate' for r in primary),
         'methods':sorted({r['method'] for r in primary}),
         'input_protocol_sha256':hashlib.sha256((ROOT/'supplement_protocol.json').read_bytes()).hexdigest()}
write_json(out/'supplement_analysis_complete.json',summary)
print(json.dumps(summary,ensure_ascii=False),flush=True)
