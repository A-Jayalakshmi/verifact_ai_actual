# VeriFact AI — FEVER-based RAG/NLI Claim Verification

**This is a real FEVER-based project implementation, not a hand-authored toy corpus.** It downloads the official FEVER training/development annotations and Wikipedia snapshot, builds an evidence sentence index, performs dense or TF-IDF retrieval, uses pretrained NLI to classify claims, and computes independent development metrics. An optional local FLAN-T5 generative explanation module is included. It is **not a pretrained end-to-end FEVER system**, and no performance is claimed until evaluation runs.

## Mac installation (Python 3.11 or 3.12 recommended)

```bash
cd /path/to/verifact_ai_actual
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.11 is unavailable, use `python3.12` if installed. On Apple Silicon, packages install natively when your Python is arm64. Check `python --version` and `uname -m`.

## Download and prepare **actual FEVER**

```bash
python prepare_fever.py --pages 3000 --dev-claims 500
```

This downloads **the original FEVER training/development annotations and the ~1.7 GB compressed Wikipedia snapshot** (several GB of disk space needed). The script scans the ZIP and selects the 3,000 most frequently referenced **training-only** Wikipedia pages, extracts their actual sentences, and samples 498 independent development claims. No dev evidence is used to select pages. Downloads may take a long time. Corpus is a real but *limited* subset; it does not claim full FEVER leaderboard comparability. Use `--pages 10000` to enlarge coverage if your Mac has sufficient RAM/storage. The `--max-sentences` limit is a safety cap; extraction stops after the shard containing the cap, so increase it for larger corpora.

**Official dataset:** https://fever.ai/dataset/fever.html . FEVER original labels: SUPPORTS, REFUTES, NOT ENOUGH INFO. Original Wikipedia snapshot: June 2017.

## Run application

```bash
python -m streamlit run app.py
```

Open http://localhost:8501 . On first use, pretrained model weights download from Hugging Face. The FAISS index is cached in `artifacts/` and reused for unchanged corpus.

## Real evaluation and baseline

```bash
python evaluate_cli.py --limit 100 --k 5 --retriever both
```

Writes `artifacts/evaluation_semantic.json` and `artifacts/evaluation_tfidf.json` with actual predictions, accuracy, macro-F1, gold evidence-set recall@k, latency and confusion matrices. Evaluate more claims with `--limit 498`. This is **not** the official FEVER score, because the system indexes a restricted training-selected page corpus, uses a simplified single-sentence NLI decision and does not implement multi-hop evidence composition. FEVER evidence-set recall counts a hit only if *all* sentences of at least one annotated evidence set are retrieved. NEI claims have no gold evidence and are excluded from retrieval recall denominator. Labels in dev are independent of training claims, but source articles may overlap.

## Architecture

`prepare_fever.py` → `data/corpus.jsonl` + `data/dev_cases.jsonl` → `SentenceTransformer(all-MiniLM-L6-v2)` + FAISS (or TF-IDF) → top-k Wikipedia sentences → `cross-encoder/nli-deberta-v3-small` NLI → conservative aggregation → verdict + quoted evidence/source URLs → optional `google/flan-t5-small` grounded-generation attempt. Generative text is **not guaranteed faithful** and does not decide the verdict.

## Limitations and research integrity

- Requires online model downloads on first run and a **large real FEVER Wikipedia ZIP**.
- CPU-first, may be slow; build/indexing could take substantial time.
- Corpus selection from training-only evidence creates limited coverage and nonrepresentative recall; never present results as full-FEVER benchmark performance.
- NLI entailment scores are not calibrated confidence in truth; thresholds 0.75 and 0.10 margin are initial heuristics, not tuned.
- No live web retrieval; evidence reflects historical Wikipedia.
- FEVER claims can require multi-sentence reasoning; this system scores individual sentences.
- NLI/generative models may produce wrong decisions or hallucinated explanations; citations link to the current Wikipedia page, which may differ from 2017 content.
- Wikipedia pages may have text formatting artifacts; raw FEVER Wikipedia lines are used.
- For an accurate report, run evaluation and include dataset size, coverage, settings, metrics and limitations.

## Files

- `prepare_fever.py`: official FEVER downloader and sentence corpus builder
- `verifact/core.py`: embedding index, TF-IDF baseline, NLI, explanations, metrics
- `app.py`: Streamlit dashboard
- `evaluate_cli.py`: repeatable command-line comparison
- `tests/`: small logic tests (`python -m pytest -q`)
- `data/`: generated on preparation, not prepackaged due to original dataset size

## Attribution

Thorne, J., Vlachos, A., Christodoulopoulos, C., & Mittal, A. (2018). FEVER: a Large-scale Dataset for Fact Extraction and VERification. NAACL-HLT. https://fever.ai/dataset/fever.html . Follow the dataset and Wikipedia licensing requirements when redistributing any extracted data.
