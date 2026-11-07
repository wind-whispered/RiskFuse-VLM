"""Editable AI drafts with explicit human confirmation and source-bound scoring."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill, Protection
from openpyxl.worksheet.datavalidation import DataValidation

from ..data.datasets import load_chair500
from ..utils import write_json, write_jsonl

TASK_HEADERS=['任务编号','来源哈希','模型','图像编号','原始生成描述','就绪状态','AI来源','预测声明数','AI候选数','复核者','复核日期','已逐句检查漏项','任务复核完成','备注','复核进度']
CLAIM_HEADERS=['任务编号','来源哈希','参考编号','候选来源','AI类型','AI主体','AI属性或关系','AI客体','AI限定与否定','AI原文','人工处理','最终类型','最终主体','最终属性或关系','最终客体','最终限定与否定','最终原文','备注']
MATCH_HEADERS=['任务编号','来源哈希','预测编号','预测类型','预测主体','预测属性或关系','预测客体','预测原文','AI建议参考编号','人工最终匹配编号','匹配已人工核对','备注']

def folder(cfg):
    path=cfg.path('review')/'reviewer_revision/assisted_human_audit'
    path.mkdir(parents=True,exist_ok=True)
    return path

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def records(path):
    if not path.exists():return []
    # A live generator appends one JSON record at a time; only newline-committed
    # records are eligible for an annotation package.
    data=path.read_bytes();lines=data.splitlines(keepends=True)
    if lines and not lines[-1].endswith((b'\n',b'\r')):lines.pop()
    return [json.loads(line) for line in lines if line.strip()]

def selected_tasks(cfg):
    samples=load_chair500(cfg)
    seed=int(cfg.get('seed',42))
    selected=sorted(samples,key=lambda s:hashlib.sha256(f'{seed}|human_audit|{s.image_id}'.encode()).hexdigest())[:int(cfg['reviewer_revision']['annotation_images_per_backbone'])]
    tasks=[]
    for backbone in cfg.backbone_names:
        processed=cfg.path('processed')
        claim_path=processed/f'claims_{backbone}_chair500.jsonl'
        completed=claim_path.exists() and claim_path.with_name(claim_path.name+'.complete.json').exists()
        if not completed:claim_path=claim_path.with_name(claim_path.name+'.partial')
        by_image=defaultdict(list)
        for row in records(claim_path):by_image[str(row['image_id'])].append(row)
        generated={str(r['image_id']):r for r in records(processed/f'generations_{backbone}_chair500.jsonl')}
        progress_path=processed/f'claims_{backbone}_chair500.jsonl.progress.json'
        checkpoint=set()
        if progress_path.exists():
            checkpoint={str(v) for v in json.loads(progress_path.read_text(encoding='utf-8')).get('completed_image_ids',[])}
        for sample in selected:
            image_id=str(sample.image_id);rows=by_image.get(image_id,[]);generation=generated.get(image_id,{})
            captions={r['caption'] for r in rows if r.get('caption') is not None}
            if generation.get('caption') is not None:captions.add(generation['caption'])
            if len(captions)>1:raise ValueError(f'Caption provenance conflict: {backbone}/{image_id}')
            caption=next(iter(captions),None)
            ready=bool(str(caption or '').strip()) and (completed or image_id in checkpoint or generation.get('n_extracted_claims')==0)
            predictions=[{k:r.get(k) for k in ('claim_id','claim_type','object','attribute','subject','relation','sentence','sentence_idx')} for r in rows if r.get('claim_type') in ('attr','rel')]
            raw_claims=generation.get('extracted_claims')
            raw_complete=isinstance(raw_claims,list) and generation.get('n_extracted_claims')==len(raw_claims)
            prediction_source=str(claim_path.relative_to(cfg.root))
            if raw_complete:
                raw_predictions=[]
                for j,claim in enumerate(raw_claims):
                    if claim.get('claim_type') not in ('attr','rel'):continue
                    raw_predictions.append(dict(claim_id=f'{backbone}|chair500|{image_id}|{j}',
                        **{k:claim.get(k) for k in ('claim_type','object','attribute','subject','relation')},
                        sentence=claim.get('sentence',caption),sentence_idx=claim.get('sentence_idx',0)))
                if predictions and predictions!=raw_predictions:
                    raise ValueError(f'Committed raw extraction and claim checkpoint disagree: {backbone}/{image_id}')
                if not predictions:
                    predictions=raw_predictions
                    prediction_source=f'generations_{backbone}_chair500.jsonl:extracted_claims'
                ready=bool(str(caption or '').strip())
            task_id=hashlib.sha256(f'{seed}|{backbone}|{image_id}'.encode()).hexdigest()[:16]
            identity={'task_id':task_id,'backbone':backbone,'image_id':image_id,'caption':caption,'predictions':predictions}
            tasks.append(dict(identity,task_sha256=digest(identity) if ready else '',image_path=sample.image_path,
                              ready=ready,caption_source=generation.get('source','formal_claim_cache' if rows else 'pending'),
                              prediction_source=prediction_source))
    return tasks

def display(claim):
    kind=claim.get('claim_type','')
    return [kind,claim.get('object','') if kind=='attr' else claim.get('subject',''),
            claim.get('attribute','') if kind=='attr' else claim.get('relation',''),
            '' if kind=='attr' else claim.get('object',''),claim.get('modality',''),claim.get('sentence','')]

def norm(value):
    return re.sub(r'\s+',' ',str(value or '').replace('_',' ').strip().lower())

def candidates(task,out):
    override=out/'ai_preannotations'/f"{task['task_id']}.json"
    if override.exists():
        data=json.loads(override.read_text(encoding='utf-8'))
        if data.get('task_sha256')!=task['task_sha256']:raise ValueError('AI preannotation refers to a changed source caption')
        return data['candidates'],data.get('source','AI assistant preannotation')
    # These are editable drafts from the evaluated parser, explicitly disclosed
    # as a shared source. They are never scored as reference truth without review.
    proposed=[];seen=set()
    for claim in task['predictions']:
        shown=display(claim);key=tuple(norm(x) for x in shown[:4])
        if key in seen:continue
        seen.add(key)
        proposed.append(dict(candidate_id=f'c{len(proposed)+1:02d}',**claim))
    return proposed,'原抽取模型候选；同源AI预标注，尚未人工复核'

def sheet_rows(book,name,key_columns):
    if name not in book:return {}
    rows=list(book[name].iter_rows(values_only=True))
    if not rows:return {}
    # Returned workbooks may rename editable columns. These aliases only read
    # the explicit actions/confirmations supplied in the returned file; they do
    # not fill a missing judgement or convert an AI proposal into a match.
    aliases={'最终匹配编号':'人工最终匹配编号','匹配已核对':'匹配已人工核对',
             'AI处理':'人工处理','AI最终匹配编号':'人工最终匹配编号',
             '匹配已人工确认':'匹配已人工核对'}
    headers=[aliases.get(h,h) for h in rows[0]];output={}
    if len(set(headers))!=len(headers):raise ValueError(f'Ambiguous duplicate columns in {name}')
    for values in rows[1:]:
        row=dict(zip(headers,values))
        if not row.get('任务编号'):continue
        key=tuple(str(row.get(k) or '') for k in key_columns)
        if key in output:raise ValueError(f'Duplicate annotation row in {name}: {key}')
        output[key]=row
    return output

def human_touched(name,row):
    keys={'任务复核':['复核者','复核日期','已逐句检查漏项','任务复核完成','备注'],
          '声明复核':['人工处理','备注'], '匹配复核':['人工最终匹配编号','匹配已人工核对','备注']}[name]
    return any(row.get(k) not in (None,'') for k in keys)

def review_fingerprint(identifier,review,claims,matches):
    reviewed_on=review.get('复核日期')
    if isinstance(reviewed_on,datetime):reviewed_on=reviewed_on.date().isoformat()
    elif isinstance(reviewed_on,date):reviewed_on=reviewed_on.isoformat()
    payload={'task_id':identifier,'task_sha256':review.get('来源哈希'),
             'review_date':str(reviewed_on or ''),'omissions_checked':review.get('已逐句检查漏项'),
             'complete':review.get('任务复核完成'),'claims':{},'matches':{}}
    for key,row in claims.items():
        if key[0]==identifier and row.get('人工处理'):
            payload['claims'][key[1]]=[str(row.get(h) or '') for h in CLAIM_HEADERS[4:17]]
    for key,row in matches.items():
        if key[0]==identifier:
            payload['matches'][key[1]]=[str(row.get(h) or '') for h in ('人工最终匹配编号','匹配已人工核对')]
    return digest(payload)

def style_sheet(ws,headers,widths):
    ws.append(headers);ws.freeze_panes='E2';ws.auto_filter.ref=f'A1:{ws.cell(1,len(headers)).column_letter}1'
    for cell in ws[1]:
        cell.font=Font(name='Microsoft YaHei',bold=True,color='FFFFFF')
        cell.fill=PatternFill('solid',fgColor='244062');cell.alignment=Alignment(vertical='top',wrap_text=True)
    for column,width in widths.items():ws.column_dimensions[column].width=width

def export_review_workbook(cfg):
    out=folder(cfg);tasks=selected_tasks(cfg)
    target=cfg.abspath(cfg['reviewer_revision']['assisted_review_workbook']);target.parent.mkdir(parents=True,exist_ok=True)
    prior={}
    if target.exists():
        archive=out/'export_history';archive.mkdir(exist_ok=True)
        version=hashlib.sha256(target.read_bytes()).hexdigest()
        preserved=archive/(version+'.xlsx')
        if not preserved.exists():shutil.copy2(target,preserved)
        old=load_workbook(target,data_only=False)
        for name,keys in [('任务复核',['任务编号']),('声明复核',['任务编号','参考编号']),('匹配复核',['任务编号','预测编号'])]:
            prior[name]=sheet_rows(old,name,keys)
        old.close()
    book=Workbook();guide=book.active;guide.title='填写说明'
    instructions=[
        ('本包性质','AI辅助、人工复核；共50张固定共享图像×4模型=200个任务。待生成的任务暂不可填写。'),
        ('第一步','在“任务复核”的“复核进度”筛选“待复核”，只处理新增任务。逐句阅读完整描述，核对属性、关系，包括错误或不确定的命题。'),
        ('第二步','在“声明复核”处理每条AI候选：保留、修改或删除。修改时改“最终”列；漏掉的命题在预留行填写并选新增。可复制新增行，但参考编号须在任务内唯一。'),
        ('第三步','在“匹配复核”将预测与最终参考一对一对应。可采用AI建议但必须填写人工最终匹配编号并标“是”；不匹配写“无匹配”。重复预测只能匹配一次。'),
        ('第四步','填真实复核者和日期，确认“已逐句检查漏项=是”和“任务复核完成=是”。AI字段和来源哈希不要修改。'),
        ('判定范围','只判断文本说了什么，不判断命题是否符合图片。图片事实及输出流畅性另评；虚假的图像命题也可能被准确抽取。'),
        ('属性范围','显式颜色、材质、状态、数量等修饰；裸对象存在不计属性。“two red cars”可拆数量和颜色两条。'),
        ('关系范围','空间、比较、包含、动作／语义关系；保留主客体、方向、否定与可能性。并列事实拆分，文本中相同命题去重。'),
        ('AI初稿来源','默认候选来自本次实际VLM抽取，属于同源、非独立的AI初稿；其中可能有错抽、重复及漏抽。请勿只确认候选而不通读全文。'),
        ('不确定处理','含义不清时在备注说明；无法完成一对一语义判断的任务暂不标完成。文本中的possibly、may等限定应保留。'),
        ('更新与保存','直接编辑本工作簿并保存。补齐模型输出时程序保留已有人工填写；已复核任务输入变化时停止覆盖。可先完成当前可复核部分。'),
        ('指标门槛','只有200个固定任务全部就绪、真实人工确认并通过一对一匹配规则后才计算整批抽取指标；未复核候选不自动视为正确。'),
    ]
    style_sheet(guide,['项目','说明'],{'A':22,'B':115})
    for row in instructions:guide.append(row)
    task_ws=book.create_sheet('任务复核');claim_ws=book.create_sheet('声明复核');match_ws=book.create_sheet('匹配复核')
    style_sheet(task_ws,TASK_HEADERS,{'A':20,'B':15,'C':16,'D':13,'E':100,'F':16,'G':42,'H':12,'I':12,'J':18,'K':16,'L':20,'M':18,'N':35,'O':16})
    style_sheet(claim_ws,CLAIM_HEADERS,{'A':20,'B':15,'C':12,'D':28,'E':10,'F':22,'G':25,'H':22,'I':20,'J':65,'K':14,'L':12,'M':22,'N':25,'O':22,'P':20,'Q':65,'R':30})
    style_sheet(match_ws,MATCH_HEADERS,{'A':20,'B':15,'C':45,'D':12,'E':22,'F':25,'G':22,'H':65,'I':20,'J':22,'K':22,'L':30})
    exported=[];ai_drafts=[];completed_ids=[];needs_review=[]
    for task in tasks:
        proposal,source=candidates(task,out) if task['ready'] else ([], '待模型输出')
        key=(task['task_id'],);old=prior.get('任务复核',{}).get(key,{})
        judgement_started=any(old.get(k) not in (None,'') for k in ('复核日期','已逐句检查漏项','任务复核完成'))
        if old and judgement_started and str(old.get('来源哈希') or '')!=task['task_sha256']:
            raise ValueError(f"Reviewed task source changed; existing workbook preserved: {task['task_id']}")
        task_row=dict(zip(TASK_HEADERS,[task['task_id'],task['task_sha256'],task['backbone'],task['image_id'],task['caption'],'可复核' if task['ready'] else '待模型输出',source,len(task['predictions']) if task['ready'] else None,len(proposal) if task['ready'] else None,'','','','','']))
        for name in ('复核者','复核日期','已逐句检查漏项','任务复核完成','备注'):task_row[name]=old.get(name,task_row.get(name,''))
        finished=task['ready'] and task_row.get('任务复核完成')=='是' and task_row.get('已逐句检查漏项')=='是'
        task_row['复核进度']='已完成' if finished else '待复核' if task['ready'] else '待生成'
        if finished:completed_ids.append(task['task_id'])
        elif task['ready']:needs_review.append(task)
        task_ws.append([task_row.get(h) for h in TASK_HEADERS])
        task_ws.row_dimensions[task_ws.max_row].height=min(300,max(75,(len(task['caption'] or '')//100+1)*14)) if task['ready'] else 24
        if not task['ready']:continue
        reference=[];known=set()
        for candidate in proposal:
            identifier=candidate['candidate_id']
            if identifier in known:raise ValueError('Duplicate AI candidate identity')
            known.add(identifier);shown=display(candidate)
            ref_key=(task['task_id'],identifier);previous=prior.get('声明复核',{}).get(ref_key,{})
            row=dict(zip(CLAIM_HEADERS,[task['task_id'],task['task_sha256'],identifier,source,*shown,'',*shown,'']))
            if previous and human_touched('声明复核',previous):
                if previous.get('来源哈希')!=task['task_sha256']:raise ValueError('Reviewed candidate source changed')
                if [str(previous.get(h) or '') for h in CLAIM_HEADERS[4:10]] != [str(v or '') for v in shown]:
                    raise ValueError('AI draft changed after human review; existing workbook preserved: '+str(ref_key))
                for h in CLAIM_HEADERS[10:]:row[h]=previous.get(h,row[h])
            claim_ws.append([row.get(h) for h in CLAIM_HEADERS]);reference.append(candidate)
        # Keep all previously added rows. Two blank slots per caption are merely
        # a convenience; the reviewer can add any number of missed propositions.
        additions={k[1]:r for k,r in prior.get('声明复核',{}).items() if k[0]==task['task_id'] and k[1] not in known}
        for identifier in ('add01','add02'):additions.setdefault(identifier,{})
        for identifier,previous in sorted(additions.items()):
            if previous and human_touched('声明复核',previous) and previous.get('来源哈希')!=task['task_sha256']:raise ValueError('Human added claim refers to changed text')
            row={h:'' for h in CLAIM_HEADERS};row.update({'任务编号':task['task_id'],'来源哈希':task['task_sha256'],'参考编号':identifier,'候选来源':'人工补充漏项'})
            for h in CLAIM_HEADERS[10:]:row[h]=previous.get(h,row[h])
            claim_ws.append([row[h] for h in CLAIM_HEADERS])
        for prediction in task['predictions']:
            shown=display(prediction);suggested=[r['candidate_id'] for r in reference if [norm(x) for x in display(r)[:4]]==[norm(x) for x in shown[:4]]]
            suggestion=suggested[0] if len(suggested)==1 else ''
            match_key=(task['task_id'],prediction['claim_id']);previous=prior.get('匹配复核',{}).get(match_key,{})
            row=dict(zip(MATCH_HEADERS,[task['task_id'],task['task_sha256'],prediction['claim_id'],*shown[:4],shown[5],suggestion,'','','']))
            if previous and human_touched('匹配复核',previous):
                if previous.get('来源哈希')!=task['task_sha256']:raise ValueError('Human match refers to changed predictions')
                for h in MATCH_HEADERS[9:]:row[h]=previous.get(h,row[h])
            match_ws.append([row.get(h) for h in MATCH_HEADERS])
        exported.append(task)
        ai_drafts.append({'task_id':task['task_id'],'task_sha256':task['task_sha256'],'source':source,'candidates':proposal,'human_reviewed':False})
    for ws,columns,formula in [(task_ws,['L','M'],'"是,否"'),(claim_ws,['K'],'"保留,修改,删除,新增"'),(claim_ws,['L'],'"attr,rel"'),(match_ws,['K'],'"是,否"')]:
        validation=DataValidation(type='list',formula1=formula,allow_blank=True);validation.errorTitle='请选择列表中的值';validation.error='请使用下拉选项。';validation.showErrorMessage=True;ws.add_data_validation(validation)
        for col in columns:validation.add(f'{col}2:{col}{max(ws.max_row+500,1000)}')
    for ws in book:
        ws.sheet_view.zoomScale=85
        ws.auto_filter.ref=ws.dimensions
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font=Font(name='Microsoft YaHei',size=10)
                cell.alignment=Alignment(vertical='top',wrap_text=True)
        if ws.title in ('任务复核','声明复核','匹配复核'):
            start={'任务复核':10,'声明复核':11,'匹配复核':10}[ws.title]
            end=14 if ws.title=='任务复核' else ws.max_column
            for row in ws.iter_rows(min_row=2,min_col=start,max_col=end):
                for cell in row:cell.fill=PatternFill('solid',fgColor='FFF2CC')
    temporary=target.with_name(target.stem+'.updating.xlsx')
    book.save(temporary);temporary.replace(target)
    write_jsonl(out/'tasks.jsonl',tasks);write_jsonl(out/'ai_preannotation_snapshot.jsonl',ai_drafts)
    snapshot=digest(ai_drafts)
    archive=out/'export_history';archive.mkdir(exist_ok=True)
    immutable=archive/(snapshot+'.ai_drafts.jsonl')
    if not immutable.exists():shutil.copy2(out/'ai_preannotation_snapshot.jsonl',immutable)
    report={'status':'complete' if len(exported)==len(tasks) else 'pending_generated_outputs','annotation_mode':'AI-assisted, human-reviewed',
            'required_human_reviewers':1,'n_shared_images':len({t['image_id'] for t in tasks}),'n_caption_tasks':len(tasks),
            'n_ready_for_human_review':len(exported),'n_tasks_awaiting_outputs':len(tasks)-len(exported),'workbook':str(target),
            'annotations_filled':False,'metrics_computed':False,'selection':'unchanged_label_score_and_action_blind_hash',
            'source_dependence':'Default draft candidates come from the evaluated extractor; not an independent or blinded reference.'}
    report['n_caption_tasks_with_codex_text_drafts']=sum('Codex' in d['source'] for d in ai_drafts)
    report['annotations_filled']=any(human_touched('任务复核',r) for r in prior.get('任务复核',{}).values())
    report['n_marked_human_complete']=len(completed_ids)
    report['n_ready_awaiting_human_review']=len(needs_review)
    write_json(out/'current_review_batch.json',{'status':'ready_for_human_review' if needs_review else 'waiting_for_more_model_outputs',
               'sample_size_retained':len(tasks),'completed_task_ids':completed_ids,
               'ready_pending_task_ids':[t['task_id'] for t in needs_review],
               'ready_pending_by_backbone':{backbone:sum(t['backbone']==backbone for t in needs_review) for backbone in cfg.backbone_names}})
    write_json(out/'sample_manifest.json',report)
    return report

def score_review_workbook(cfg):
    if int(cfg['reviewer_revision'].get('annotation_required_reviewers',1))!=1:
        raise ValueError('This AI-assisted protocol is configured for one human review per task; use a separately specified protocol for multiple independent reviews.')
    out=folder(cfg);target=cfg.abspath(cfg['reviewer_revision']['assisted_review_workbook'])
    tasks=selected_tasks(cfg)
    if not target.exists():return {'status':'pending_human_annotations','metrics_computed':False}
    book=load_workbook(target,data_only=False)
    reviews=sheet_rows(book,'任务复核',['任务编号'])
    claims=sheet_rows(book,'声明复核',['任务编号','参考编号'])
    matches=sheet_rows(book,'匹配复核',['任务编号','预测编号'])
    attestation_path=out/'returned_review/project_owner_attestation.json'
    attestation=json.loads(attestation_path.read_text(encoding='utf-8')) if attestation_path.exists() else {}
    workbook_signature=hashlib.sha256(target.read_bytes()).hexdigest()
    attested_workbook=attestation.get('workbook_sha256')==workbook_signature
    pending=[];complete=[];counts=defaultdict(lambda:{'true_positive':0,'n_predicted':0,'n_reference':0})
    audit=[]
    for task in tasks:
        identifier=task['task_id'];review=reviews.get((identifier,),{})
        if not task['ready'] or review.get('任务复核完成')!='是':pending.append(identifier);continue
        if review.get('来源哈希')!=task['task_sha256']:raise ValueError('Confirmed task hash changed: '+identifier)
        reviewer=str(review.get('复核者') or '').strip()
        reviewer_provenance='workbook'
        preserved_review=attestation.get('completed_review_signatures',{}).get(identifier)==review_fingerprint(identifier,review,claims,matches)
        attested_task=identifier in attestation.get('completed_task_ids',[]) and (attested_workbook or preserved_review)
        if not reviewer and attested_task:
            reviewer=attestation.get('anonymous_reviewer_identifier','')
            reviewer_provenance='project_owner_confirmation; personal_identity_not_recorded'
        reviewed_on=review.get('复核日期')
        confirmation_date=attestation.get('date_received') if attested_task else None
        if not reviewer or review.get('已逐句检查漏项')!='是' or (not reviewed_on and not confirmation_date):
            raise ValueError('Human review provenance and full omission check are required: '+identifier)
        if isinstance(reviewed_on,(date,datetime)):reviewed_on=reviewed_on.date().isoformat() if isinstance(reviewed_on,datetime) else reviewed_on.isoformat()
        try:date.fromisoformat(str(reviewed_on or confirmation_date))
        except ValueError:raise ValueError('Use an actual ISO review date YYYY-MM-DD: '+identifier)
        reference={};actions=[]
        original_candidates,_=candidates(task,out)
        original_candidates={r['candidate_id']:r for r in original_candidates}
        for key,row in claims.items():
            if key[0]!=identifier:continue
            action=row.get('人工处理');is_draft=bool(row.get('AI类型'))
            if not action:
                if is_draft:raise ValueError('AI candidate has not been reviewed: '+str(key))
                if any(row.get(h) for h in CLAIM_HEADERS[11:17]):raise ValueError('Added reference needs an explicit human action: '+str(key))
                continue
            if row.get('来源哈希')!=task['task_sha256']:raise ValueError('Reference hash mismatch: '+str(key))
            if key[1] in original_candidates:
                expected=display(original_candidates[key[1]])
                if [str(row.get(h) or '') for h in CLAIM_HEADERS[4:10]] != [str(v or '') for v in expected]:
                    raise ValueError('Edit the final-reference columns, preserving the original AI draft: '+str(key))
            if action not in ('保留','修改','删除','新增'):raise ValueError('Unknown review action: '+str(key))
            if action=='删除':actions.append({'candidate_id':key[1],'action':action});continue
            use_ai=action=='保留'
            values=[row.get(h) for h in (CLAIM_HEADERS[4:10] if use_ai else CLAIM_HEADERS[11:17])]
            kind,subject,predicate,obj,modality,sentence=values
            if kind not in ('attr','rel') or not str(subject or '').strip() or not str(predicate or '').strip() or (kind=='rel' and not str(obj or '').strip()):
                raise ValueError('Reference proposition is incomplete: '+str(key))
            if not str(sentence or '').strip():raise ValueError('Reference needs supporting caption text: '+str(key))
            reference[key[1]]={'claim_type':kind,'subject':subject,'predicate':predicate,'object':obj,'modality':modality,'sentence':sentence}
            actions.append({'candidate_id':key[1],'action':action,'reference':reference[key[1]]})
        predictions={p['claim_id']:p for p in task['predictions']};used=set();task_counts=defaultdict(lambda:{'true_positive':0,'n_predicted':0,'n_reference':0})
        checked_matches=[]
        for predicted_id,prediction in predictions.items():
            row=matches.get((identifier,predicted_id),{})
            if row.get('匹配已人工核对')!='是' or row.get('来源哈希')!=task['task_sha256']:
                raise ValueError('A prediction match has not been human-confirmed: '+predicted_id)
            ref_id=str(row.get('人工最终匹配编号') or '').strip()
            if not ref_id:raise ValueError('Use a reference id or 无匹配: '+predicted_id)
            kind=prediction['claim_type'];task_counts[kind]['n_predicted']+=1
            if ref_id!='无匹配':
                if ref_id not in reference or ref_id in used or reference[ref_id]['claim_type']!=kind:
                    raise ValueError('Human semantic matching must be one-to-one and same-type: '+predicted_id)
                used.add(ref_id);task_counts[kind]['true_positive']+=1
            checked_matches.append({'predicted_claim_id':predicted_id,'reference_id':None if ref_id=='无匹配' else ref_id})
        for ref in reference.values():task_counts[ref['claim_type']]['n_reference']+=1
        for kind,values in task_counts.items():
            for group in ((task['backbone'],kind),('ALL',kind),('ALL','ALL')):
                for key,value in values.items():counts[group][key]+=value
        complete.append(identifier)
        audit.append({'task_id':identifier,'task_sha256':task['task_sha256'],'reviewer':reviewer,'review_date':str(reviewed_on) if reviewed_on else None,
                      'owner_confirmation_date':confirmation_date,
                      'review_date_provenance':'workbook' if reviewed_on else 'individual review date not recorded; owner completion confirmation dated separately',
                      'reviewer_provenance':reviewer_provenance,
                      'image_id':task['image_id'],'backbone':task['backbone'],'actions':actions,'matches':checked_matches,'counts':dict(task_counts)})
    book.close()
    # Intermediate progress is allowed; partial selected outputs are not used
    # to publish accuracy for the fixed 200-caption audit.
    report={'status':'pending_human_annotations' if pending else 'complete','metrics_computed':False,
            'n_expected_tasks':len(tasks),'n_human_confirmed_tasks':len(complete),'n_pending_tasks':len(pending),
            'annotation_mode':'AI-assisted, human-reviewed','independent_human_annotation':False,
            'reviewers':sorted({r['reviewer'] for r in audit}),
            'workbook_sha256':workbook_signature,
            'reviewer_identity_note':'Anonymous reviewer token supplied under the project owner’s explicit confirmation when no reviewer name is present in the returned workbook.'}
    ai_snapshot=out/'ai_preannotation_snapshot.jsonl'
    report['ai_draft_snapshot_sha256']=hashlib.sha256(ai_snapshot.read_bytes()).hexdigest() if ai_snapshot.exists() else None
    write_jsonl(out/'confirmed_human_review_records.jsonl',audit)
    if pending:
        write_json(out/'review_progress.json',report);return report
    metrics=[]
    for backbone in [*cfg.backbone_names,'ALL']:
        for kind in ('attr','rel'):counts[(backbone,kind)]
    for (backbone,kind),values in sorted(counts.items()):
        tp=values['true_positive'];p=values['n_predicted'];g=values['n_reference']
        metrics.append({'backbone':backbone,'claim_type':kind,**values,'precision':tp/p if p else None,'recall':tp/g if g else None,'f1':2*tp/(p+g) if p+g else None})
    import numpy as np
    image_ids=sorted({r['image_id'] for r in audit})
    draws=np.random.default_rng(int(cfg.get('seed',42))).integers(0,len(image_ids),(int(cfg['reviewer_revision'].get('annotation_bootstrap_repetitions',1000)),len(image_ids)))
    for metric in metrics:
        matrix=[]
        for image in image_ids:
            vector={k:0 for k in ('true_positive','n_predicted','n_reference')}
            for record in audit:
                if record['image_id']!=image or (metric['backbone']!='ALL' and record['backbone']!=metric['backbone']):continue
                for kind,values in record['counts'].items():
                    if metric['claim_type']!='ALL' and metric['claim_type']!=kind:continue
                    for key in vector:vector[key]+=values[key]
            matrix.append([vector[k] for k in ('true_positive','n_predicted','n_reference')])
        summed=np.asarray(matrix,dtype=float)[draws].sum(axis=1)
        with np.errstate(divide='ignore',invalid='ignore'):
            measures={'precision':summed[:,0]/summed[:,1],'recall':summed[:,0]/summed[:,2],
                      'f1':2*summed[:,0]/(summed[:,1]+summed[:,2])}
        for name,values in measures.items():
            usable=values[np.isfinite(values)]
            interval=np.quantile(usable,[.025,.975]).tolist() if len(usable) else [None,None]
            metric[name+'_image_bootstrap_95ci']=interval
        metric['n_shared_image_units']=len(image_ids)
    report.update(metrics_computed=True,extraction=metrics,
                  uncertainty='Percentile image bootstrap, preserving the shared image across all four backbones; ratios use pooled proposition counts.',
                  limitation=f"AI-assisted reference reviewed by {len(report['reviewers'])} human reviewer(s), with shared-source initial candidates; no independent or blinded human annotation claim. Textual extraction only; visual factual support and fluency excluded.")
    write_json(out/'assisted_human_audit_results.json',report)
    write_json(out/'review_progress.json',report)
    manifest_path=out/'sample_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest.update(n_marked_human_complete=len(complete),n_ready_awaiting_human_review=0,
                        annotations_filled=True,metrics_computed=True,
                        returned_workbook_sha256=workbook_signature)
        write_json(manifest_path,manifest)
    write_json(out/'current_review_batch.json',dict(status='all_human_reviews_complete',
               sample_size_retained=len(tasks),completed_task_ids=complete,
               ready_pending_task_ids=[],ready_pending_by_backbone={b:0 for b in cfg.backbone_names}))
    return report
