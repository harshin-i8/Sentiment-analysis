import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Load model and tokenizer
MODEL_DIR = "saved_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# Define emotion labels (adjust if needed based on your training)
EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'love', 'surprise']

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_label = torch.argmax(probs, dim=1).item()
    return EMOTION_LABELS[predicted_label], probs[0][predicted_label].item()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Emotion Decoder", layout="wide")

st.title("🧠 Emotion Decoder from Social Media Conversations")
st.markdown("Analyze the emotional tone of social media texts using a deep learning model.")

# Background styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stTextArea {
            font-size: 18px;
        }
    </style>
""", unsafe_allow_html=True)

# Input Section
st.header("📝 Enter Social Media Text")
user_input = st.text_area("Paste a tweet, comment, or post here 👇", height=200)

if st.button("🔍 Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        emotion, confidence = predict_emotion(user_input)
        st.success(f"### Detected Emotion: `{emotion.upper()}`")
        st.progress(int(confidence * 100))

        # Emoji mapping (optional for a creative touch)
        emoji_map = {
            'joy': '😄',
            'sadness': '😢',
            'anger': '😠',
            'fear': '😨',
            'love': '❤️',
            'surprise': '😲'
        }
        st.markdown(f"## {emoji_map.get(emotion, '❓')} Emotion: **{emotion}** with {confidence:.2%} confidence")

# Footer
st.markdown("---")
st.markdown("Made with 💙 using Streamlit & Transformers")

