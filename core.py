import json, re, time
from pathlib import Path
from urllib.parse import quote
import numpy as np

LABELS = ['SUPPORTS','REFUTES','NOT ENOUGH INFO']

def clean(text):
    return (text or '').replace('-LRB-', '(').replace('-RRB-', ')').replace('-LSB-', '[').replace('-RSB-', ']').replace(' -COLON- ', ': ').replace(' -COMMA- ', ', ').strip()

def source_url(page):
    return 'https://en.wikipedia.org/wiki/' + quote(page.replace(' ', '_'), safe='()_')

def read_corpus(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

class Retriever:
    def __init__(self, corpus, model_name='sentence-transformers/all-MiniLM-L6-v2', index_dir='artifacts', batch_size=64):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.corpus = corpus
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device='cpu')
        self.index_dir = Path(index_dir); self.index_dir.mkdir(parents=True, exist_ok=True)
        self.faiss = faiss
        import hashlib
        signature = hashlib.sha256(('\n'.join(x['id'] + x['text'] for x in corpus) + model_name).encode()).hexdigest()[:16]
        self.path = self.index_dir / ('index_' + signature + '.faiss')
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
        else:
            self.index = faiss.IndexFlatIP(self.model.get_sentence_embedding_dimension())
            for i in range(0, len(corpus), batch_size):
                vectors = self.model.encode([x['text'] for x in corpus[i:i+batch_size]], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype('float32')
                self.index.add(vectors)
            faiss.write_index(self.index, str(self.path))

    def search(self, claim, k=5):
        if not self.corpus: return []
        vec = self.model.encode([claim], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        scores, ids = self.index.search(vec, min(k, len(self.corpus)))
        return [{**self.corpus[int(idx)], 'retrieval_score':float(score)} for score, idx in zip(scores[0],ids[0]) if idx>=0]

class TFIDFRetriever:
    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.corpus=corpus; self.vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english',max_features=120000)
        self.matrix=self.vectorizer.fit_transform([x['text'] for x in corpus])
    def search(self, claim, k=5):
        from sklearn.metrics.pairwise import linear_kernel
        sim=linear_kernel(self.vectorizer.transform([claim]),self.matrix).ravel()
        indices=np.argsort(sim)[::-1][:k]
        return [{**self.corpus[int(i)],'retrieval_score':float(sim[i])} for i in indices]

class Verifier:
    def __init__(self, model_name='cross-encoder/nli-deberta-v3-small'):
        from transformers import pipeline
        self.pipe=pipeline('text-classification',model=model_name,tokenizer=model_name,device=-1,top_k=None)
        labels=self.pipe.model.config.id2label
        self.label_map={i: str(v).lower() for i,v in labels.items()}
        if not any('entail' in v for v in self.label_map.values()) or not any('contrad' in v for v in self.label_map.values()):
            raise ValueError(f'NLI model has unsupported labels: {self.label_map}')

    def infer(self, claim, evidence):
        if not evidence: return {'label':'NOT ENOUGH INFO','evidence':[],'strength':0.0}
        # Hugging Face text-classification supports sentence pairs via tokenizer directly.
        tokenizer=self.pipe.tokenizer; model=self.pipe.model
        import torch
        pairs=[(e['text'],claim) for e in evidence]
        outputs=[]
        for start in range(0,len(pairs),8):
            part=pairs[start:start+8]
            batch=tokenizer([a for a,b in part],[b for a,b in part],return_tensors='pt',truncation=True,padding=True,max_length=384)
            with torch.inference_mode(): probs=torch.softmax(model(**batch).logits,dim=-1).cpu().numpy()
            outputs.extend(probs)
        enriched=[]
        for e,prob in zip(evidence,outputs):
            vals={self.label_map[i]:float(prob[i]) for i in range(len(prob))}
            ent=max((v for k,v in vals.items() if 'entail' in k),default=0.)
            con=max((v for k,v in vals.items() if 'contrad' in k),default=0.)
            neu=max((v for k,v in vals.items() if 'neutral' in k),default=0.)
            enriched.append({**e,'entailment':ent,'contradiction':con,'neutral':neu})
        best_sup=max(enriched,key=lambda e:e['entailment']); best_ref=max(enriched,key=lambda e:e['contradiction'])
        # Conservative thresholds: NLI scores are not calibrated truth probabilities.
        if best_ref['contradiction']>=0.75 and best_ref['contradiction']>best_sup['entailment']+0.10:
            label='REFUTES'; selected=best_ref; strength=best_ref['contradiction']
        elif best_sup['entailment']>=0.75 and best_sup['entailment']>best_ref['contradiction']+0.10:
            label='SUPPORTS'; selected=best_sup; strength=best_sup['entailment']
        else:
            label='NOT ENOUGH INFO'; selected=max(enriched,key=lambda e:max(e['entailment'],e['contradiction'])); strength=max(selected['entailment'],selected['contradiction'])
        return {'label':label,'evidence':enriched,'selected':selected,'strength':strength}

def explain(claim, result):
    e=result.get('selected')
    if not e: return 'No relevant evidence was retrieved from the indexed corpus. This does not prove the claim is false.'
    if result['label']=='SUPPORTS': return f'The retrieved passage from {e["page"]} supports the claim: “{e["text"]}”'
    if result['label']=='REFUTES': return f'The retrieved passage from {e["page"]} contradicts the claim: “{e["text"]}”'
    return f'The indexed evidence does not establish whether this claim is true or false. Closest passage ({e["page"]}): “{e["text"]}”'

def generate_explanation(claim, result, model_name='google/flan-t5-small'):
    # Optional generative RAG explanation; not used to determine the verdict.
    from transformers import pipeline
    gen=pipeline('text2text-generation',model=model_name,device=-1)
    excerpts=result.get('evidence',[])[:3]
    if not excerpts: return explain(claim,result)
    context='\n'.join(f'[{i+1}] {e["page"]}: {e["text"]}' for i,e in enumerate(excerpts))
    prompt=(f'Explain the verification decision using ONLY the evidence. If the evidence is insufficient, say so. '
            f'Do not invent facts or sources. Claim: {claim}\nVerdict: {result["label"]}\nEvidence:\n{context}\nExplanation:')
    draft=gen(prompt,max_new_tokens=110,do_sample=False)[0]['generated_text'].strip()
    # Provide source excerpts alongside the generated text; generation may hallucinate.
    return draft or explain(claim,result)

def evaluate(cases, retriever, verifier, k=5, limit=None):
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    output=[]
    for case in cases[:limit] if limit else cases:
        start=time.perf_counter(); retrieved=retriever.search(case['claim'],k)
        pred=verifier.infer(case['claim'],retrieved)
        gold_sets=[set(s) for s in case.get('evidence_sets',[]) if s]
        retrieved_ids={x['id'] for x in retrieved}
        hit=any(s.issubset(retrieved_ids) for s in gold_sets) if gold_sets else None
        output.append({'id':case['id'],'claim':case['claim'],'gold':case['label'],'predicted':pred['label'], 'evidence_hit':hit,'seconds':time.perf_counter()-start})
    if not output: raise ValueError('No evaluation cases available')
    y=[x['gold'] for x in output]; p=[x['predicted'] for x in output]
    hits=[x['evidence_hit'] for x in output if x['evidence_hit'] is not None]
    metrics={'n':len(output),'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=LABELS,average='macro',zero_division=0), 'evidence_set_recall_at_k':float(np.mean(hits)) if hits else None,'mean_latency_seconds':float(np.mean([x['seconds'] for x in output])), 'classification_report':classification_report(y,p,labels=LABELS,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y,p,labels=LABELS).tolist(),'labels':LABELS}
    return metrics,output
