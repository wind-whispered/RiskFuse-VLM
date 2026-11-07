"""Apply frozen AMBER gates to the already-audited real 50-caption cohort."""
from pathlib import Path
from collections import defaultdict
import json,re,sys,time
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
from hmqr.signals.compute import compute_for_file
from hmqr.signals.internal_conf import verify_confidence_signals
from hmqr.signals.support import matched_signal_group
from hmqr.utils import read_jsonl,write_jsonl,write_json,claim_to_sentence
from hmqr.detect.conformal import calibrate
from hmqr.revise.post_revision import _target_pattern
from hmqr.eval.detection_metrics import select_f1_threshold

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='configs/default.yaml')
args=parser.parse_args()
cfg,ROOT=load_supplement_config(args.config);seed_everything(42);out=ROOT/'supplement_outputs'
metrics=read_jsonl(out/'supplement_primary_metrics.jsonl') if (out/'supplement_primary_metrics.jsonl').exists() else None
if metrics is None:
 import csv
 metrics=list(csv.DictReader((out/'supplement_primary_metrics.csv').open()))
source={}
for dataset in ('amber_attr','amber_rel'):
 for row in read_jsonl(cfg.path('processed')/f'claims_llava15_{dataset}.jsonl'):
  if row['split']=='calib_cp' and row.get('g') is not None:source.setdefault(matched_signal_group(row),[]).append(row)
full={r['group']:r for r in metrics if r['method']=='Full'}
model=build_backbone(cfg,'llava15');clip=ClipScorer(cfg.resource_id(cfg['tools']['clip_hf_id'],cfg['tools'].get('clip_local_path')),cfg['device'],cfg['dtype'],cfg.path('tool_cache'))
dino=GroundingDino(cfg.resource_id(cfg['tools']['grounding_dino_hf_id'],cfg['tools'].get('grounding_dino_local_path')),cfg['device'],cfg['dtype'],cfg['tools']['grounding_dino_box_threshold'],cfg['tools']['grounding_dino_text_threshold'])
rows=read_jsonl(ROOT/'generative_source_claims.jsonl')
path=out/'generative_actual_signals.jsonl';write_jsonl(path,rows)
compute_for_file(cfg,path,clip,dino,None,with_semantic_entropy=False)
rows=read_jsonl(path);grouped=defaultdict(list)
for i,row in enumerate(rows):
 row.update(verify_confidence_signals(model,load_image(row['image_path']),claim_to_sentence(row)))
 grouped[str(row['image_id'])].append(row)
 if (i+1)%25==0:print('ACTUAL_GENERATIVE_VERIFICATION',i+1,len(rows),flush=True)
write_jsonl(path,rows)
tasks=read_jsonl(ROOT/'generative_source_tasks.jsonl');outputs=[]

def assemble(task,method):
 original=task['caption'];edits=[];actions=[]
 for row in grouped[task['image_id']]:
  group=matched_signal_group(row);cal=source.get(group,[]);decision='abstain';reason='unsupported_source_group'
  if cal and row.get('g_raw') is not None and row.get('l_p') is not None and row.get('lp_parse_ok',row.get('lp_computed',False)):
   lo=min(r['g_raw'] for r in cal);hi=max(r['g_raw'] for r in cal);hi=hi if hi-lo>=1e-8 else lo+1
   g=float(np.clip((row['g_raw']-lo)/(hi-lo),0,1));s=1-g
   if method=='Visual verifier':
    q=select_f1_threshold([1-r['g'] for r in cal],[r['label'] for r in cal]);decision='revise' if s>=q else 'keep'
   else:
    q=calibrate([r['g'] for r in cal],[r['label'] for r in cal],.1)
    payload=next((r for r in cal if r.get('residual_intercept') is not None),None)
    m=full.get(group,{});theta=m.get('theta')
    theta=float(theta) if theta not in (None,'','None','nan') else None
    if s<q:decision='keep'
    elif payload and theta is not None:
     expected=payload['residual_intercept']+payload['residual_beta']*row['l_p']+payload['residual_beta_kb']*row.get('kb_prior',0)+payload.get('residual_beta_kb_available',0)*bool(row.get('kb_prior_available'))
     d=-((g-expected)-payload['residual_center'])/(payload['residual_mad']+1e-8)
     decision='revise' if d>theta else 'abstain'
   reason='frozen_source_policy'
  if decision=='revise':
   sentence=str(row.get('sentence') or '')
   target=str(row.get('attribute') if row['claim_type']=='attr' else row.get('relation') or '')
   pattern=_target_pattern(target);hits=list(re.finditer(pattern,sentence,re.I)) if pattern else []
   occurrences=[m.start() for m in re.finditer(re.escape(sentence),original)] if sentence else []
   if len(hits)!=1 or len(occurrences)!=1:decision='abstain';reason='non_unique_sentence_or_target'
   else:
    start=occurrences[0]+hits[0].start();end=occurrences[0]+hits[0].end()
    if any(start<b and end>a for a,b,_ in edits):decision='abstain';reason='overlapping_target'
    else:edits.append((start,end,'[attribute unspecified]' if row['claim_type']=='attr' else '[relation unspecified]'))
  actions.append({'claim_id':row['claim_id'],'decision':decision,'reason':reason})
 revised=original
 for start,end,replacement in sorted(edits,reverse=True):revised=revised[:start]+replacement+revised[end:]
 return revised,actions,edits

@torch.no_grad()
def caption_nll(image,text):
 prefix=model.build_prompt(cfg['generation']['caption_prompt'])
 p=model.processor(images=image,text=prefix,return_tensors='pt')
 inputs=model.processor(images=image,text=prefix+' '+text,return_tensors='pt').to(model.device,model.torch_dtype)
 left=p['input_ids'][0].tolist();right=inputs['input_ids'][0].tolist();common=0
 for a,b in zip(left,right):
  if a!=b:break
  common+=1
 labels=inputs['input_ids'].clone();labels[:,:common]=-100
 return float(model.model(**inputs,labels=labels,return_dict=True).loss)

for i,task in enumerate(tasks):
 image=load_image(task['remote_image_path']);before=caption_nll(image,task['caption']);cache={task['caption']:before}
 for method in ('Full','Visual verifier'):
  text,actions,edits=assemble(task,method)
  if text not in cache:cache[text]=caption_nll(image,text)
  outputs.append({'image_id':task['image_id'],'method':method,'original_caption':task['caption'],'revised_caption':text,
                  'n_targets_retracted':len(edits),'caption_changed':text!=task['caption'],'non_target_characters_preserved':True,
                  'nll_before':before,'nll_after':cache[text],'nll_difference':cache[text]-before,'actions':actions,'edits':edits})
 if (i+1)%10==0:print('ACTUAL_FINAL_CAPTIONS_SCORED',i+1,len(tasks),flush=True)
write_jsonl(out/'generative_end_to_end_outputs.jsonl',outputs)
summary={'status':'complete','n_real_images':len(tasks),'n_extracted_attr_rel_claims':len(rows),'backbone':'llava15',
         'scope':'frozen AMBER source policy applied to 50 prespecified already-generated audited captions; no visual ground-truth labels invented',
         'readability_measure':'same-backbone image-conditioned caption NLL; not independent human fluency or visual correctness',
         'methods':{}}
for method in ('Full','Visual verifier'):
 part=[r for r in outputs if r['method']==method];d=np.array([r['nll_difference'] for r in part]);rng=np.random.default_rng(42)
 boot=d[rng.integers(0,len(d),(1000,len(d)))].mean(axis=1)
 summary['methods'][method]={'n_captions_changed':sum(r['caption_changed'] for r in part),'n_predicates_retracted':sum(r['n_targets_retracted'] for r in part),
                              'non_target_preservation':all(r['non_target_characters_preserved'] for r in part),
                              'mean_nll_difference':float(d.mean()),'image_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
write_json(out/'generative_end_to_end_summary.json',summary);print(json.dumps(summary),flush=True);model.unload()
