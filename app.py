
import streamlit as st
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    mcq_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
    mcq_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")
    return summarizer, mcq_tokenizer, mcq_model

summarizer, mcq_tokenizer, mcq_model = load_models()

def fetch_news(api_key, limit=5):
    url = 'https://api.currentsapi.services/v1/latest-news'
    headers = {'Authorization': api_key}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data.get('news', [])[:limit]

def summarize_text(text):
    text = text[:1500]
    if len(text) < 40:
        return "Not enough content to summarize."
    summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]['summary_text']

def generate_mcq(summary):
    prompt = f"""
    Create a multiple choice question based on this news summary.
    Include four answer options, and mark the correct one with **:

    {summary}
    """
    input_ids = mcq_tokenizer(prompt, return_tensors="pt").input_ids
    outputs = mcq_model.generate(input_ids, max_length=256, do_sample=True, temperature=0.7)
    result = mcq_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

st.set_page_config(page_title="News MCQ Generator", layout="centered")
st.title("📰 News-Based MCQ Generator")
st.markdown("Generate summaries and questions from **real-time news** using AI.")

api_key = st.text_input("🔑 Enter your Currents API Key")

if api_key:
    with st.spinner("Fetching and processing news..."):
        articles = fetch_news(api_key, limit=5)

    if not articles:
        st.error("❌ No articles found. Check your API key or quota.")
    else:
        for i, article in enumerate(articles):
            title = article.get("title", "No Title")
            desc = article.get("description", "No Description")
            full_text = f"{title}. {desc}"

            st.markdown(f"### 📰 Article {i+1}: {title}")
            st.markdown("**📖 Description:**")
            st.write(desc)

            summary = summarize_text(full_text)
            st.markdown("**📝 Summary:**")
            st.success(summary)

            mcq = generate_mcq(summary)
            st.markdown("**❓ MCQ:**")
            st.info(mcq)

            st.markdown("---")
