import streamlit as st
import os
import sys

# Add project root
if "src" not in os.listdir():
    PROJECT_ROOT = os.path.abspath("..")
else:
    PROJECT_ROOT = os.path.abspath(".")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.conversation.chatbot import ChatBot

st.set_page_config(page_title="NewsBot Intelligence System", layout="wide")

st.title("📰 NewsBot Intelligence System – Web Demo")
st.write("Analyze news using advanced NLP: summarization, sentiment, NER, translation, similarity & more.")

bot = ChatBot()

# Sidebar
st.sidebar.header("NewsBot Controls")
mode = st.sidebar.selectbox(
    "Choose Analysis Mode:",
    [
        "Chat",
        "Summarization",
        "Sentiment",
        "NER",
        "Translation",
        "Semantic Similarity",
        "Classification"
    ]
)

st.sidebar.write("---")

# Main UI
if mode == "Chat":
    st.header("💬 Chat with NewsBot")

    user_input = st.text_input("Enter your message:")
    if st.button("Send"):
        if user_input.strip():
            response = bot.ask(user_input)
            st.write("### 🤖 Response:")
            st.write(response)

elif mode == "Summarization":
    st.header("📝 Summarize News Article")
    text = st.text_area("Paste article text here:")
    if st.button("Summarize"):
        result = bot.ask("Summarize this news:", {"text": text})
        st.write("### Summary:")
        st.write(result)

elif mode == "Sentiment":
    st.header("😊 Sentiment Analysis")
    text = st.text_area("Enter text:")
    if st.button("Analyze Sentiment"):
        result = bot.ask("What is the sentiment of this?", {"text": text})
        st.write("### Sentiment Result:")
        st.write(result)

elif mode == "NER":
    st.header("🔎 Named Entity Recognition")
    text = st.text_area("Enter text:")
    if st.button("Extract Entities"):
        result = bot.ask("Extract the entities from this news", {"text": text})
        st.write("### Entities Detected:")
        st.write(result)

elif mode == "Translation":
    st.header("🌍 Translation & Language Detection")
    text = st.text_area("Enter text in Spanish or English:")
    if st.button("Translate to English"):
        result = bot.ask("Translate this to English", {"text": text})
        st.write("### Translation:")
        st.write(result)

elif mode == "Semantic Similarity":
    st.header("🔗 Compare Semantic Similarity Between Two Texts")
    text1 = st.text_area("Text A:")
    text2 = st.text_area("Text B:")
    if st.button("Compare Similarity"):
        result = bot.ask("Are these two similar?", {"text": text1, "reference": text2})
        st.write("### Similarity Score:")
        st.write(result)

elif mode == "Classification":
    st.header("🏷 News Topic Classification")
    text = st.text_area("Enter article text:")
    if st.button("Classify"):
        result = bot.ask("Classify the topic of this article.", {"text": text})
        st.write("### Predicted Topic:")
        st.write(result)
