"""Complete missing text anchors using the existing entity-token recipe."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from supplement_common import load_supplement_config
from hmqr.config import load_config
from hmqr.backbones.registry import build_backbone
from hmqr.utils import read_jsonl,write_json
from hmqr.data.images import load_image
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='configs/default.yaml')
args=parser.parse_args()
cfg,ROOT=load_supplement_config(args.config);out=ROOT/'supplement_outputs'
features={}
for i in range(3):features.update(json.loads((out/f'features_shard_{i}.json').read_text()))
rows=[r for i in range(3) for dataset in ('amber_attr','amber_rel') for r in read_jsonl(out/f'shard_{i}_{dataset}.jsonl')]
missing=[r for r in rows if not features.get(r['claim_id'],{}).get('file')]
result={};model=build_backbone(cfg,'llava15') if missing else None
for i,row in enumerate(missing):
    words=[str(w).replace('_',' ') for w in (row.get('subject'),row.get('object')) if w]
    state=model.forward_states(load_image(row['image_path']),cfg['generation']['caption_prompt'],row['sentence'],keywords=words,need_attentions=False)
    hidden=state['hidden_states'];vectors=[]
    for word in words:
        span=state['text_token_spans'].get(word)
        if span is not None:vectors.append(hidden[-1,span[0]:span[1],:].mean(axis=0))
    if not vectors:continue
    name=hashlib.sha256(row['claim_id'].encode()).hexdigest()+'.npy'
    np.save(out/name,np.mean(vectors,axis=0).astype(np.float16))
    result[row['claim_id']]={'file':name,'recipe':'fixed_final_layer_aligned_entity_anchor_for_unaligned_canonical_predicate','layer':int(hidden.shape[0]-1)}
    if (i+1)%50==0:print('MISSING_FEATURE_ANCHORS_COMPLETED',i+1,len(missing),flush=True)
write_json(out/'features_repaired.json',result)
if model:model.unload()
print('FEATURE_ANCHOR_COMPLETION',len(result),'of',len(missing),flush=True)
