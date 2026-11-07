"""Actual prespecified supplementary cohort; no software smoke/tests."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from supplement_common import load_supplement_config
from hmqr.config import load_config,seed_everything
from hmqr.backbones.registry import build_backbone
from hmqr.tools.clip_scorer import ClipScorer
from hmqr.tools.grounding_dino import GroundingDino
from hmqr.signals.compute import compute_for_file
from hmqr.data.images import load_image
from hmqr.utils import read_jsonl,write_jsonl,write_json

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True);a=p.parse_args()
cfg,ROOT=load_supplement_config(a.config);seed_everything(cfg.get('seed',42))
out=ROOT/'supplement_outputs';out.mkdir(exist_ok=True)
start=time.perf_counter()
model=build_backbone(cfg,'llava15')
clip=ClipScorer(cfg.resource_id(cfg['tools']['clip_hf_id'],cfg['tools'].get('clip_local_path')),cfg['device'],cfg['dtype'],cfg.path('tool_cache'))
dino=GroundingDino(cfg.resource_id(cfg['tools']['grounding_dino_hf_id'],cfg['tools'].get('grounding_dino_local_path')),cfg['device'],cfg['dtype'],cfg['tools']['grounding_dino_box_threshold'],cfg['tools']['grounding_dino_text_threshold'],cfg.path('tool_cache'))
features={};summaries=[]
for dataset in ('amber_attr','amber_rel'):
    source=read_jsonl(ROOT/'data/supplement'/f'claims_llava15_{dataset}.jsonl')
    rows=source[a.shard::a.shards]
    if not rows:continue
    path=out/f'shard_{a.shard}_{dataset}.jsonl';write_jsonl(path,rows)
    compute_for_file(cfg,path,clip,dino,model,with_semantic_entropy=False)
    rows=read_jsonl(path)
    for index,row in enumerate(rows):
        key=str(row.get('attribute') if row['claim_type']=='attr' else row.get('relation') or '').replace('_',' ')
        try:
            state=model.forward_states(load_image(row['image_path']),cfg['generation']['caption_prompt'],row['sentence'],keywords=[key],need_attentions=False)
            span=state['text_token_spans'].get(key)
            if span is None:raise ValueError('predicate token alignment unavailable')
            hidden=state['hidden_states']
            vector=hidden[-1,span[0]:span[1],:].mean(axis=0)
            name=hashlib.sha256(row['claim_id'].encode()).hexdigest()+'.npy'
            np.save(out/name,np.asarray(vector,dtype=np.float16))
            features[row['claim_id']]={'file':name,'recipe':'prespecified_final_layer_predicate_token_mean','layer':int(hidden.shape[0]-1)}
        except (ValueError,KeyError) as exc:
            features[row['claim_id']]={'unavailable':str(exc)}
        if (index+1)%20==0:print('SUPPLEMENT_FEATURES',dataset,index+1,len(rows),flush=True)
    summaries.append({'dataset':dataset,'n_rows':len(rows)})
write_json(out/f'features_shard_{a.shard}.json',features)
write_json(out/f'worker_{a.shard}_complete.json',dict(status='complete',rows=summaries,elapsed_seconds=time.perf_counter()-start))
model.unload()
