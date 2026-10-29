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
    def __init__(
        self,
        model_name="cross-encoder/nli-deberta-v3-small",
        support_threshold=0.75,
        refute_threshold=0.85,
        margin=0.15,
    ):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.eval()

        self.support_threshold = support_threshold
        self.refute_threshold = refute_threshold
        self.margin = margin

        self.label_map = {
            int(i): str(label).lower()
            for i, label in self.model.config.id2label.items()
        }

        def find_label(keyword):
            matches = [
                i for i, label in self.label_map.items()
                if keyword in label
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Cannot uniquely identify {keyword}: {self.label_map}"
                )
            return matches[0]

        self.entail_idx = find_label("entail")
        self.contra_idx = find_label("contrad")
        self.neutral_idx = find_label("neutral")

    def _score_pairs(self, pairs):
        outputs = []

        for start in range(0, len(pairs), 8):
            part = pairs[start:start + 8]

            batch = self.tokenizer(
                [premise for premise, hypothesis in part],
                [hypothesis for premise, hypothesis in part],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=384,
            )

            with self.torch.inference_mode():
                logits = self.model(**batch).logits
                probabilities = self.torch.softmax(
                    logits, dim=-1
                ).cpu().numpy()

            for prob in probabilities:
                outputs.append({
                    "entailment": float(prob[self.entail_idx]),
                    "contradiction": float(prob[self.contra_idx]),
                    "neutral": float(prob[self.neutral_idx]),
                })

        return outputs

    def infer(self, claim, evidence):
        if not evidence:
            return {
                "label": "NOT ENOUGH INFO",
                "evidence": [],
                "selected": None,
                "strength": 0.0,
            }

        # First score each evidence sentence independently.
        pairs = [(e["text"], claim) for e in evidence]
        scores = self._score_pairs(pairs)

        enriched = [
            {**e, **score}
            for e, score in zip(evidence, scores)
        ]

        # A simple lexical-overlap check, independent of the
        # retriever's score scale.
        import re

        stopwords = {
            "the", "a", "an", "is", "was", "were", "are",
            "of", "in", "on", "at", "to", "for", "and",
            "or", "by", "with", "that", "this", "it",
            "has", "have", "had", "be", "been"
        }

        def content_words(text):
            return {
                w for w in re.findall(r"\b\w+\b", text.lower())
                if w not in stopwords and len(w) > 2
            }

        claim_words = content_words(claim)

        for e in enriched:
            evidence_words = content_words(
                e["text"] + " " + e.get("page", "").replace("_", " ")
            )

            e["word_overlap"] = (
                len(claim_words & evidence_words)
                / max(len(claim_words), 1)
            )

        # Additional evidence combinations:
        # Only combine sentences from the same article.
        grouped = {}
        for e in enriched:
            grouped.setdefault(e.get("page", ""), []).append(e)

        combined_candidates = []

        for page, items in grouped.items():
            if len(items) < 2:
                continue

            # Limit combinations to avoid large NLI costs.
            top_items = sorted(
                items,
                key=lambda e: (
                    e["entailment"],
                    e["word_overlap"]
                ),
                reverse=True
            )[:3]

            for i in range(len(top_items)):
                for j in range(i + 1, len(top_items)):
                    first = top_items[i]
                    second = top_items[j]

                    combined_candidates.append({
                        "page": page,
                        "text": first["text"] + " " + second["text"],
                        "id": f'{first.get("id", i)}+{second.get("id", j)}',
                        "retrieval_score": max(
                            first.get("retrieval_score", 0.0),
                            second.get("retrieval_score", 0.0)
                        ),
                        "combined": True,
                    })

        if combined_candidates:
            combined_scores = self._score_pairs([
                (e["text"], claim)
                for e in combined_candidates
            ])

            for e, score in zip(
                combined_candidates, combined_scores
            ):
                e.update(score)

                evidence_words = content_words(
                    e["text"] + " " + e.get("page", "").replace("_", " ")
                )

                e["word_overlap"] = (
                    len(claim_words & evidence_words)
                    / max(len(claim_words), 1)
                )

            enriched.extend(combined_candidates)

        # Require modest lexical anchoring before accepting
        # a confident verdict. This is an experimental safeguard,
        # not a validated measure of semantic relevance.
        eligible = [
            e for e in enriched
            if e["word_overlap"] >= 0.20
        ]

        if not eligible:
            selected = max(
                enriched,
                key=lambda e: max(
                    e["entailment"], e["contradiction"]
                )
            )

            return {
                "label": "NOT ENOUGH INFO",
                "evidence": enriched,
                "selected": selected,
                "strength": max(
                    selected["entailment"],
                    selected["contradiction"]
                ),
            }

        best_support = max(
            eligible,
            key=lambda e: e["entailment"]
        )

        best_refute = max(
            eligible,
            key=lambda e: e["contradiction"]
        )

        support_score = best_support["entailment"]
        refute_score = best_refute["contradiction"]

        if (
            support_score >= self.support_threshold
            and support_score > refute_score + self.margin
        ):
            label = "SUPPORTS"
            selected = best_support
            strength = support_score

        elif (
            refute_score >= self.refute_threshold
            and refute_score > support_score + self.margin
        ):
            label = "REFUTES"
            selected = best_refute
            strength = refute_score

        else:
            label = "NOT ENOUGH INFO"
            selected = max(
                eligible,
                key=lambda e: max(
                    e["entailment"], e["contradiction"]
                )
            )
            strength = max(
                selected["entailment"],
                selected["contradiction"]
            )

        return {
            "label": label,
            "evidence": enriched,
            "selected": selected,
            "strength": strength,
        }

def explain(claim, result):
    e=result.get('selected')
    if not e: return 'No relevant evidence was retrieved from the indexed corpus. This does not prove the claim is false.'
    if result['label']=='SUPPORTS': return f'The retrieved passage from {e["page"]} supports the claim: “{e["text"]}”'
    if result['label']=='REFUTES': return f'The retrieved passage from {e["page"]} contradicts the claim: “{e["text"]}”'
    return f'The indexed evidence does not establish whether this claim is true or false. Closest passage ({e["page"]}): “{e["text"]}”'



def generate_explanation(claim, result, model_name='google/flan-t5-small'):
    # Never generate unsupported factual explanations when evidence is insufficient.
    if result['label'] == 'NOT ENOUGH INFO':
        return (
            "The retrieved evidence is insufficient to verify this claim. "
            "No factual explanation can be reliably generated from the "
            "available sources."
        )

    # Optional generative explanation; does not determine the verdict.
    from transformers import pipeline

    gen = pipeline(
        'text2text-generation',
        model=model_name,
        device=-1
    )

    selected = result.get('selected')

    if not selected:
        return explain(claim, result)

    context = f'{selected["page"]}: {selected["text"]}'

    prompt = (
        "Explain the verification decision using ONLY the evidence below. "
        "Do not add names, dates, places, or facts absent from the evidence. "
        "If the evidence is insufficient, say so.\n"
        f"Claim: {claim}\n"
        f"Verdict: {result['label']}\n"
        f"Evidence: {context}\n"
        "Explanation:"
    )

    draft = gen(
        prompt,
        max_new_tokens=110,
        do_sample=False
    )[0]['generated_text'].strip()

    return draft or explain(claim, result)

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
