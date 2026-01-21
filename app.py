import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Path to your fine‑tuned model on Hugging Face Hub
model_path = "RamaAI/fake-news-distilbert"

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    ).to("cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# --- Streamlit UI ---
st.title("📰 Fake News Detector")
st.markdown("""
Analyze news articles in real-time using AI-powered detection to identify potential misinformation.
⚠️ **Disclaimer**: This tool is trained on a labeled dataset. Predictions may not always match reality — always verify with credible sources.
""")

headline = st.text_input("Headline (Optional)", placeholder="Enter the news headline...")
news_text = st.text_area("News Text *", placeholder="Paste or type the news article text here...")

if st.button("Analyze"):
    full_text = (headline + " " + news_text).strip() if headline else news_text.strip()

    if not full_text:
        st.warning("⚠️ Please enter text before analyzing.")
    else:
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True)
        if inputs['input_ids'].shape[1] == 0:
            st.warning("⚠️ Empty input — nothing to analyze.")
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(probs, dim=1).detach().numpy()[0]
                confidence = probs[0][pred].detach().numpy().item()

            label = "🔴 Potential Fake News" if pred == 0 else "🟢 Real News"

            if pred == 0:
                st.error(f"{label} (Confidence: {confidence:.2%})")
            else:
                st.success(f"{label} (Confidence: {confidence:.2%})")

            if confidence >= 0.9:
                st.success("✅ High confidence in prediction.")
            elif confidence >= 0.7:
                st.info("ℹ️ Moderate confidence — likely real, but worth verifying.")
            else:
                st.warning("⚠️ Low confidence — prediction uncertain.")
