import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


st.set_page_config(page_title="Sentiment Analysis", layout="wide")
st.title("LLM Sentiment & Attention Analysis Dashboard")
st.subheader("First Project")
st.write("This application visualizes both the classification predictions of the DistilBERT model and analyzes the underlying syntactic & semantic attention weights (XAI) in real-time.")


@st.cache_resource
def load_models():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    classifier = pipeline("sentiment-analysis", model=model_name)
    return tokenizer, model, classifier

tokenizer, model, classifier = load_models()


st.divider()
user_input = st.text_input("Enter an English sentence to analyze:", "The movie was not bad, but the sound system was terrible.")

if st.button("Run Model & Analyze", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a valid sentence!")
    else:
       
        st.markdown("###  Model Prediction")
        with st.spinner("Generating prediction..."):
            result = classifier(user_input)[0]
            label = result['label']
            score = result['score'] * 100
            
            if label == "POSITIVE":
                st.success(f"**Prediction:** {label} (Confidence Score: {score:.2f}%)")
            else:
                st.error(f"**Prediction:** {label} (Confidence Score: {score:.2f}%)")
        
        st.info(" **Error Analysis Note:** Compare the Syntactic vs Semantic attention weights below to evaluate if the model falls into local token traps or successfully captures logical contrasts.")
        
        st.divider()
        st.markdown("###  Dual Attention Head Analysis (XAI)")
        
        
        inputs = tokenizer(user_input, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            attention = outputs.attentions
        
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        
        col1, col2 = st.columns(2)
        
       # SYNTACTİC HEAD 
        with col1:
            st.markdown("####  Syntactic Head (Layer 0, Head 2)")
            st.caption("Focuses on basic grammatical structures, positional weights, and adjacent tokens.")
            
            syn_attention = attention[0][0][2] # Layer 0, Head 2
            syn_cls_scores = syn_attention[0].tolist()
            syn_pairs = list(zip(tokens, syn_cls_scores))
            syn_pairs.sort(key=lambda x: x[1], reverse=True)
            
            st.write("Top 5 attended tokens by `[CLS]`:")
            for i, (token, score) in enumerate(syn_pairs[:5], 1):
                st.write(f"**{i}. {token}** — {score * 100:.2f}%")
                st.progress(min(max(score, 0.0), 1.0))
                
        # SEMANTIC HEAD
        with col2:
            st.markdown("#### Semantic Head (Layer 5, Head 0)")
            st.caption("Focuses on deeper contextual meaning, logical contrasts, and sentiment carriers.")
            
            sem_attention = attention[5][0][0] # Layer 5, Head 0
            sem_cls_scores = sem_attention[0].tolist()
            sem_pairs = list(zip(tokens, sem_cls_scores))
            sem_pairs.sort(key=lambda x: x[1], reverse=True)
            
            st.write("Top 5 attended tokens by `[CLS]`:")
            for i, (token, score) in enumerate(sem_pairs[:5], 1):
                st.write(f"**{i}. {token}** — {score * 100:.2f}%")
                st.progress(min(max(score, 0.0), 1.0))
