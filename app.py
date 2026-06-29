import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .stAppDeployButton {display: none !important;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


st.title(" LLM Sentiment & Attention Analysis Dashboard")
st.subheader(" Proje No 1")
st.write("This application visualizes classification predictions of the DistilBERT model and allows interactive exploration of all underlying attention layers and heads in real-time.")


@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
    return tokenizer, model

tokenizer, model = load_model()


user_input = st.text_input("Enter an English sentence to analyze:", "The movie was not bad, but the sound system was terrible.")


st.markdown("---")
st.sidebar.header(" Attention Explorer Controls")
st.sidebar.write("Farklı katman (layer) ve başlıkları (head) buradan özgürce karşılaştırabilirsiniz.")

st.sidebar.subheader("Left Visualization Panel")
layer_left = st.sidebar.slider("Select Left Layer", min_value=0, max_value=5, value=0, step=1)
head_left = st.sidebar.slider("Select Left Head", min_value=0, max_value=11, value=2, step=1)

st.sidebar.subheader("Right Visualization Panel")
layer_right = st.sidebar.slider("Select Right Layer", min_value=0, max_value=5, value=5, step=1)
head_right = st.sidebar.slider("Select Right Head", min_value=0, max_value=11, value=5, step=1)

if st.button("Run Model & Analyze", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a valid sentence.")
    else:
        
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs)
        
        logits = outputs.logits
        attentions = outputs.attentions 
        
       
        probs = torch.softmax(logits, dim=1).tolist()[0]
        prediction_id = torch.argmax(logits, dim=1).item()
        labels = ["NEGATIVE", "POSITIVE"]
        pred_label = labels[prediction_id]
        confidence = probs[prediction_id] * 100
        
        
        st.subheader(" Model Sentiment Prediction")
        if pred_label == "POSITIVE":
            st.success(f"Prediction: **{pred_label}** (Confidence Score: **{confidence:.2f}%**)")
        else:
            st.error(f"Prediction: **{pred_label}** (Confidence Score: **{confidence:.2f}%**)")
            
        st.info("💡 **Error Analysis Note:** Compare the Syntactic vs Semantic attention weights below to evaluate if the model falls into local token traps or successfully captures logical contrasts.")
        
       
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
       
        attention_matrix_left = attentions[layer_left][0, head_left, :, :]
        cls_attention_left = attention_matrix_left[0, :].tolist()
        
       
        attention_matrix_right = attentions[layer_right][0, head_right, :, :]
        cls_attention_right = attention_matrix_right[0, :].tolist()
        
      
        df_left = pd.DataFrame({"Token": tokens, "Attention": cls_attention_left}).sort_values(by="Attention", ascending=False).head(5)
        df_right = pd.DataFrame({"Token": tokens, "Attention": cls_attention_right}).sort_values(by="Attention", ascending=False).head(5)
        
        
        st.markdown("---")
        st.subheader("Interactive Multi-Layer Attention Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### Left Panel (Layer {layer_left}, Head {head_left})")
            st.write("Visualizing the selected layer configuration weights for `[CLS]` token:")
            for _, row in df_left.iterrows():
                st.write(f"**{row['Token']}** — {row['Attention']*100:.2f}%")
                st.progress(float(row['Attention']))
                
        with col2:
            st.markdown(f"### Right Panel (Layer {layer_right}, Head {head_right})")
            st.write("Visualizing the selected layer configuration weights for `[CLS]` token:")
            for _, row in df_right.iterrows():
                st.write(f"**{row['Token']}** — {row['Attention']*100:.2f}%")
                st.progress(float(row['Attention']))
