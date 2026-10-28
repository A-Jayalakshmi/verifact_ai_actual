"""Download official FEVER and construct a training-page-selected, sentence-level evidence index.
Requires ~2 GB download and free disk space. Dev claims never select corpus pages.
"""
import argparse, json, random, re, zipfile, urllib.request
from pathlib import Path
from collections import Counter
from verifact.core import clean
BASE='https://fever.ai/download/fever/'

def download(url,dest):
    if dest.exists() and dest.stat().st_size>0:
        print('Using',dest); return
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix(dest.suffix+'.part')
    print('Downloading',url,'to',dest,'(large downloads can take time)')
    urllib.request.urlretrieve(url,tmp)
    tmp.replace(dest)

def load_claims(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def evidence_sets(row):
    sets=[]
    for group in row.get('evidence',[]):
        s=[(e[2],e[3]) for e in group if len(e)>=4 and e[2] is not None and e[3] is not None]
        if s: sets.append(s)
    return sets

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pages',type=int,default=3000,help='Number of unique training evidence pages to index')
    p.add_argument('--dev-claims',type=int,default=500,help='Independent development claims sampled across labels')
    p.add_argument('--max-sentences',type=int,default=120000)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--data-dir',default='data')
    a=p.parse_args(); d=Path(a.data_dir); d.mkdir(exist_ok=True,parents=True)
    if (d/'corpus.jsonl').exists() or (d/'dev_cases.jsonl').exists():
        raise SystemExit('Prepared data already exists. Move/delete corpus.jsonl and dev_cases.jsonl to rebuild explicitly.')
    train=d/'train.jsonl'; dev=d/'shared_task_dev.jsonl'; wiki=d/'wiki-pages.zip'
    for name,dest in [('train.jsonl',train),('shared_task_dev.jsonl',dev),('wiki-pages.zip',wiki)]: download(BASE+name,dest)
    rng=random.Random(a.seed)
    # Only training evidence determines page selection. Dev labels and evidence do not select pages.
    counts=Counter()
    for row in load_claims(train):
        for group in evidence_sets(row):
            for page,_ in group: counts[page]+=1
    selected={page for page,_ in counts.most_common(a.pages)}
    print('Selected',len(selected),'training-only evidence pages; extracting sentences from FEVER Wikipedia archive')
    sentences={}
    with zipfile.ZipFile(wiki) as z:
        members=[ name for name in z.namelist() if name.endswith(".jsonl") and not name.startswith("__MACOSX/") and not name.split("/")[-1].startswith("._")]
        for idx,member in enumerate(members):
            print(f'Wikipedia shard {idx+1}/{len(members)}: {member}',flush=True)
            with z.open(member) as fh:
                for raw in fh:
                    page=json.loads(raw)
                    pid=page.get('id','')
                    if pid not in selected: continue
                    for line in page.get('lines','').splitlines():
                        bits=line.split('\t',2)
                        if len(bits)<2 or not bits[0].isdigit(): continue
                        sid=int(bits[0]); text=clean(bits[1])
                        if len(text)<20: continue
                        key=f'{pid}::{sid}'
                        sentences[key]={'id':key,'page':pid,'sentence_id':sid,'text':text}
            if len(sentences)>=a.max_sentences: print('Reached max-sentences; stopping at shard boundary');break
    with (d/'corpus.jsonl').open('w',encoding='utf-8') as f:
        for item in sentences.values(): f.write(json.dumps(item,ensure_ascii=False)+'\n')
    grouped={k:[] for k in ['SUPPORTS','REFUTES','NOT ENOUGH INFO']}
    for row in load_claims(dev):
        if row['label'] not in grouped:continue
        gold=[[f'{page}::{sid}' for page,sid in group] for group in evidence_sets(row)]
        grouped[row['label']].append({'id':row['id'],'claim':row['claim'],'label':row['label'],'evidence_sets':gold})
    n=a.dev_claims//3
    cases=[]
    for label,items in grouped.items(): cases.extend(rng.sample(items,min(len(items),n)))
    rng.shuffle(cases)
    with (d/'dev_cases.jsonl').open('w',encoding='utf-8') as f:
        for item in cases:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    possible=[x for x in cases if x['evidence_sets']]
    coverage=sum(any(set(group).issubset(sentences) for group in x['evidence_sets']) for x in possible)
    metadata={'source':BASE,'page_selection':'most frequent pages referenced in training evidence only','pages_requested':a.pages,'sentences':len(sentences),'dev_claims':len(cases),'verifiable_dev_claims':len(possible),'gold_evidence_set_present_in_corpus':coverage,'coverage_fraction':coverage/len(possible) if possible else None,'seed':a.seed,'note':'Corpus selection limits recall; this is not a full 5.4-million-page FEVER benchmark.'}
    (d/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('COMPLETE:',json.dumps(metadata,indent=2))

if __name__=='__main__':main()
