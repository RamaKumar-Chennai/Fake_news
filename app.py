import streamlit as st
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Path to your fine‑tuned model folder (with model.safetensors or pytorch_model.bin)
model_path = r"D:\VS_CODE\Fake_news\distilbert_model"


# Load tokenizer and model safely
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(
    model_path,
    device_map=None,          # don’t auto-map
    dtype=torch.float32,      # replaces deprecated torch_dtype
    low_cpu_mem_usage=False   # ensures weights are fully loaded
)
model.eval()

# --- Streamlit UI ---
st.title("📰 Fake News Detector")
st.markdown("""
Analyze news articles in real-time using AI-powered detection to identify potential misinformation.

⚠️ **Disclaimer**: This tool is trained on a labeled dataset. Predictions may not always match reality — always verify with credible sources.
""")

headline = st.text_input("Headline (Optional)", placeholder="Enter the news headline...")
news_text = st.text_area("News Text *", placeholder="Paste or type the news article text here...")

# Clear button to reset inputs
if st.button("Clear"):
    headline = ""
    news_text = ""
    st.rerun()

# Analyze button
if st.button("Analyze"):
    full_text = (headline + " " + news_text).strip() if headline else news_text.strip()

    if not full_text:
        st.warning("⚠️ Please enter text before analyzing.")
    else:
        # Tokenize safely
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True)

        # Guard against empty tokenization
        if inputs['input_ids'].shape[1] == 0:
            st.warning("⚠️ Empty input — nothing to analyze.")
        else:
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

                # Prediction index (0 = Fake, 1 = Real)
                pred = torch.argmax(probs, dim=1).detach().cpu().numpy()[0]

                # Confidence score for the predicted class
                confidence = probs[0][pred].detach().cpu().numpy().item()

            # Dataset mapping: 0 = Fake/Misleading, 1 = Real/True
            label = "🔴 Potential Fake News" if pred == 0 else "🟢 Real News"

            st.subheader("Prediction:")
            if pred == 0:
                st.error(f"{label} (Confidence: {confidence:.2%})")
            else:
                st.success(f"{label} (Confidence: {confidence:.2%})")

            # Confidence interpretation
            if confidence >= 0.9:
                st.success("✅ High confidence in prediction.")
            elif confidence >= 0.7:
                st.info("ℹ️ Moderate confidence — likely real, but worth verifying.")
            else:
                st.warning("⚠️ Low confidence — prediction uncertain.")