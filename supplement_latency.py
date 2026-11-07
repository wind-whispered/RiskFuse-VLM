"""Fresh exclusive-GPU native cached-prior path versus generation and verifier."""
from pathlib import Path
import csv,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import torch
from supplement_common import load_supplement_config
from hmqr.config import load_config,seed_everything
from hmqr.backbones.registry import build_backbone
from hmqr.tools.clip_scorer import ClipScorer
from hmqr.tools.grounding_dino import GroundingDino
from hmqr.data.images import load_image
from hmqr.signals.global_clip import g_global
from hmqr.signals.grounding_attr import g_attr
from hmqr.signals.grounding_count import count_consistency
from hmqr.signals.support import attribute_kind,matched_signal_group
from hmqr.signals.internal_conf import verify_confidence_signals
from hmqr.revise.post_revision import target_retraction_sentence
from hmqr.detect.conformal import calibrate
from hmqr.utils import read_jsonl,write_json,write_jsonl
from hmqr.eval.detection_metrics import select_f1_threshold
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='configs/default.yaml')
args=parser.parse_args()
cfg,ROOT=load_supplement_config(args.config);seed_everything(42);out=ROOT/'supplement_outputs'
rows=read_jsonl(cfg.path('processed')/'claims_llava15_amber_attr.jsonl')
primary=list(csv.DictReader((out/'supplement_primary_metrics.csv').open()))
theta={r['group']:float(r['theta']) if r.get('theta') not in (None,'','nan','None') else None for r in primary if r['method']=='Full'}
cp={};allcp={}
for r in rows:
 if r['split']=='calib_cp' and r.get('g') is not None:
  allcp.setdefault(matched_signal_group(r),[]).append(r)
  if r.get('d_conflict') is not None and r.get('l_p') is not None:cp.setdefault(matched_signal_group(r),[]).append(r)
chosen={}
for r in rows:
 if r['split']=='test' and r.get('g_raw') is not None and matched_signal_group(r) in cp:chosen.setdefault(r['image_id'],r)
jobs=sorted(chosen.values(),key=lambda r:hashlib.sha256(('42|fresh_latency|'+r['image_id']).encode()).hexdigest())[:11]
model=build_backbone(cfg,'llava15');clip=ClipScorer(cfg.resource_id(cfg['tools']['clip_hf_id'],cfg['tools'].get('clip_local_path')),cfg['device'],cfg['dtype'],cfg.path('tool_cache'))
dino=GroundingDino(cfg.resource_id(cfg['tools']['grounding_dino_hf_id'],cfg['tools'].get('grounding_dino_local_path')),cfg['device'],cfg['dtype'],cfg['tools']['grounding_dino_box_threshold'],cfg['tools']['grounding_dino_text_threshold'])
def measured(fn):
 torch.cuda.synchronize();start=time.perf_counter();value=fn();torch.cuda.synchronize();return value,time.perf_counter()-start
def grounding(image,row):
 dino._cache.clear();kind=attribute_kind(row)
 global_score=g_global(clip,image,row)
 if kind=='state':raw=g_attr(clip,dino,image,row['object'],row['attribute'],str(row['image_path']))['g_attr']
 elif kind=='number':raw=count_consistency(dino,image,row['object'],row['attribute'],cache_key=str(row['image_path']))['g_count']
 else:raw=global_score
 return raw
records=[]
for i,row in enumerate(jobs):
 image,decode=measured(lambda:load_image(row['image_path']))
 _,generation=measured(lambda:model.generate(image,cfg['generation']['caption_prompt'],max_new_tokens=256))
 group=matched_signal_group(row);cal=cp[group];bounds=allcp[group]
 lo=min(r['g_raw'] for r in bounds);hi=max(r['g_raw'] for r in bounds);hi=hi if hi-lo>=1e-8 else lo+1
 q=calibrate([r['g'] for r in cal],[r['label'] for r in cal],.1)
 vq=select_f1_threshold([1-r['g'] for r in cal],[r['label'] for r in cal]);payload=cal[0]
 def full_path():
  raw=grounding(image,row);belief=verify_confidence_signals(model,image,row['sentence'])
  if raw is None or belief.get('l_p') is None:return 'abstain'
  g=float(np.clip((raw-lo)/(hi-lo),0,1));expected=payload['residual_intercept']+payload['residual_beta']*belief['l_p']+payload['residual_beta_kb']*row.get('kb_prior',0)+payload.get('residual_beta_kb_available',0)*bool(row.get('kb_prior_available'))
  d=-((g-expected)-payload['residual_center'])/(payload['residual_mad']+1e-8)
  return target_retraction_sentence(row) if 1-g>=q and theta.get(group) is not None and d>theta[group] else 'keep_or_abstain'
 _,full_time=measured(full_path)
 def visual_path():
  raw=grounding(image,row)
  if raw is None:return 'abstain'
  g=float(np.clip((raw-lo)/(hi-lo),0,1))
  return target_retraction_sentence(row) if 1-g>=vq else 'keep'
 _,visual_time=measured(visual_path)
 records.append({'image_id':row['image_id'],'warmup':i==0,'decode_seconds':decode,'generation_seconds':generation,'full_native_seconds':full_time,'visual_verifier_seconds':visual_time})
 print('ACTUAL_LATENCY_IMAGE',i+1,len(jobs),flush=True)
write_jsonl(out/'fresh_latency_records.jsonl',records)
measured_rows=[r for r in records if not r['warmup']]
summary={'status':'complete','backbone':'llava15','dataset':'amber_attr','n_images':len(measured_rows),'warmup_images':1,
         'hardware':torch.cuda.get_device_name(),'scope':'one native claim per image; corpus priors cached; free-form extraction and model load excluded; image decode separate; fresh visual evidence for both policies',
         'means_seconds':{k:float(np.mean([r[k] for r in measured_rows])) for k in ('decode_seconds','generation_seconds','full_native_seconds','visual_verifier_seconds')}}
write_json(out/'fresh_latency_summary.json',summary);print(json.dumps(summary),flush=True);model.unload()
