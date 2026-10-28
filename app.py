from pathlib import Path
import json
import pandas as pd
import streamlit as st
from verifact.core import read_corpus, Retriever, TFIDFRetriever, Verifier, explain, generate_explanation, evaluate, source_url

st.set_page_config(page_title='VeriFact AI | Evidence-based verification',page_icon='🔎',layout='wide')
st.title('🔎 VeriFact AI')
st.caption('Retrieval-Augmented Evidence Verification · FEVER Wikipedia corpus · NLI · Optional Generative Explanation')
DATA=Path('data')
if not (DATA/'corpus.jsonl').exists():
    st.error('The real FEVER corpus has not been prepared yet. Run `python prepare_fever.py --pages 3000 --dev-claims 500` in Terminal, then reload.')
    st.stop()
@st.cache_data(show_spinner='Loading evidence corpus…')
def corpus():return read_corpus(DATA/'corpus.jsonl')
@st.cache_resource(show_spinner='Loading Sentence Transformer and building/loading FAISS index…')
def semantic():return Retriever(corpus())
@st.cache_resource(show_spinner='Loading TF-IDF index…')
def lexical():return TFIDFRetriever(corpus())
@st.cache_resource(show_spinner='Loading NLI verification model…')
def nli():return Verifier()

meta=json.loads((DATA/'metadata.json').read_text()) if (DATA/'metadata.json').exists() else {}
with st.sidebar:
    st.header('Configuration')
    mode=st.radio('Retriever',['Semantic (FAISS)','TF-IDF'])
    k=st.slider('Top-k evidence',1,10,5)
    generative=st.toggle('Generative explanation (FLAN-T5)',value=False,help='Downloads another model. Its output is not a verified factual source.')
    st.divider()
    st.metric('Indexed evidence sentences',f'{len(corpus()):,}')
    st.metric('Selected Wikipedia pages',f'{meta.get("pages_requested",0):,}')
    st.caption('Coverage is limited to the indexed FEVER Wikipedia subset. NOT ENOUGH INFO means insufficient indexed evidence, not proof of falsehood.')

t1,t2,t3=st.tabs(['Verify a claim','Evaluation','Corpus & methodology'])
with t1:
    claim=st.text_area('Enter a factual claim',placeholder='Example: Albert Einstein was born in Germany.',height=95)
    if st.button('Verify claim',type='primary'):
        if not claim.strip(): st.warning('Please enter a claim.')
        else:
            with st.spinner('Retrieving and verifying evidence…'):
                retriever=semantic() if mode.startswith('Semantic') else lexical()
                evidence=retriever.search(claim,k)
                result=nli().infer(claim,evidence)
            labels={'SUPPORTS':'Supported','REFUTES':'Contradicted','NOT ENOUGH INFO':'Insufficient Evidence'}
            st.subheader(labels[result['label']]);st.caption('Decision from NLI thresholds; model scores are not calibrated truth probabilities.')
            st.write('**Grounded explanation:**',explain(claim,result))
            if generative:
                with st.spinner('Generating an optional grounded explanation…'):
                    st.write('**Generated explanation (check against sources):**',generate_explanation(claim,result))
            st.subheader('Retrieved evidence and NLI probabilities')
            for i,e in enumerate(result['evidence'],1):
                with st.expander(f'{i}. {e["page"].replace("_"," ")} · retrieval {e["retrieval_score"]:.3f}',expanded=i==1):
                    st.write(e['text']); st.markdown(f'[Wikipedia article]({source_url(e["page"])})')
                    st.caption(f'Entailment {e["entailment"]:.3f} | Contradiction {e["contradiction"]:.3f} | Neutral {e["neutral"]:.3f}')
with t2:
    st.subheader('Independent FEVER development evaluation')
    st.write('Development claims are distinct from training claims. Wikipedia pages were selected using training evidence only. Metrics are calculated live; no fabricated accuracy values.')
    cases_path=DATA/'dev_cases.jsonl'
    if cases_path.exists():
        cases=read_corpus(cases_path)
        st.caption(f'{len(cases)} FEVER development cases available. Evaluation may take several minutes on CPU.')
        limit=st.slider('Number of cases to evaluate',min_value=3,max_value=len(cases),value=min(30,len(cases)),step=1)
        if st.button('Run evaluation'):
            with st.spinner('Evaluating claims…'):
                ret=semantic() if mode.startswith('Semantic') else lexical()
                metrics,rows=evaluate(cases,ret,nli(),k,limit)
            st.session_state['evaluation']=(metrics,rows)
        if 'evaluation' in st.session_state:
            m,rows=st.session_state['evaluation']
            a,b,c,d=st.columns(4)
            a.metric('Accuracy',f'{m["accuracy"]:.1%}');b.metric('Macro F1',f'{m["macro_f1"]:.3f}')
            c.metric('Evidence-set recall@k','N/A' if m['evidence_set_recall_at_k'] is None else f'{m["evidence_set_recall_at_k"]:.1%}')
            d.metric('Mean latency',f'{m["mean_latency_seconds"]:.2f}s')
            st.caption('Evidence-set recall@k is calculated on verifiable cases with annotated gold evidence; limited corpus coverage lowers its ceiling.')
            st.write('**Confusion matrix (rows: actual, columns: predicted)**')
            st.dataframe(pd.DataFrame(m['confusion_matrix'],index=m['labels'],columns=m['labels']))
            st.dataframe(pd.DataFrame(rows),hide_index=True)
            st.download_button('Download evaluation JSON',json.dumps({'metrics':m,'predictions':rows},indent=2),'evaluation.json','application/json')
    else:st.warning('Run prepare_fever.py to create evaluation cases.')
with t3:
    st.subheader('Indexed FEVER Wikipedia evidence')
    st.json(meta)
    query=st.text_input('Search indexed evidence')
    if query:
        st.dataframe(pd.DataFrame(semantic().search(query,10)),hide_index=True)
    st.markdown('**Method:** training-evidence-page selection → original Wikipedia sentence extraction → SentenceTransformer/TF-IDF retrieval → NLI evidence/claim pair scoring → conservative three-way decision → cited explanation. Optional FLAN-T5 text generation is separate from verdict logic.')
    st.warning('This is a corpus-limited research prototype, not a live web fact-checker. Wikipedia evidence comes from the FEVER June 2017 snapshot. The explanation generator can still hallucinate; inspect the linked evidence.')
