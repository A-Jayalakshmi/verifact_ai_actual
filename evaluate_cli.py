import argparse,json
from pathlib import Path
from verifact.core import read_corpus,Retriever,TFIDFRetriever,Verifier,evaluate

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=100);p.add_argument('--k',type=int,default=5);p.add_argument('--retriever',choices=['semantic','tfidf','both'],default='both');a=p.parse_args()
    corpus=read_corpus('data/corpus.jsonl');cases=read_corpus('data/dev_cases.jsonl');v=Verifier()
    Path('artifacts').mkdir(exist_ok=True)
    for mode in (['semantic','tfidf'] if a.retriever=='both' else [a.retriever]):
        r=Retriever(corpus) if mode=='semantic' else TFIDFRetriever(corpus)
        m,rows=evaluate(cases,r,v,a.k,a.limit)
        dest=Path('artifacts')/f'evaluation_{mode}.json';dest.write_text(json.dumps({'metrics':m,'predictions':rows},indent=2))
        print(mode,'accuracy',round(m['accuracy'],4),'macro_f1',round(m['macro_f1'],4),'recall@k',m['evidence_set_recall_at_k'],'saved',dest)
if __name__=='__main__':main()
