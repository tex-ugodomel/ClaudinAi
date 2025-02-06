import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import openai
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

# Set OpenAI API key (ensure you set this in your environment)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI with ClaudinAi branding
st.set_page_config(page_title="ClaudinAi - RAG Chatbot", page_icon="🤖")
st.title("🤖 ClaudinAi - RAG Chatbot")

# Sidebar settings
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Choose model:", ["gpt-4o-mini", "o1-mini"])
st.sidebar.markdown("---")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        text = extract_text(pdf_path)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return None

# Function to embed text using OpenAI API
def get_embeddings(text):
    response = openai.Embedding.create(model="text-embedding-3-small", input=text)
    return response["data"][0]["embedding"]

# Upload PDF
st.subheader("📂 Upload a PDF Document")
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            st.success("✅ PDF processed successfully!")
            
            # Chunk text for embeddings
            chunks = pdf_text.split("\n\n")  # Simple chunking strategy
            embeddings = [get_embeddings(chunk) for chunk in chunks]
            
            # Save to DataFrame
            df = pd.DataFrame({"content": chunks, "embeddings": embeddings})
            df.to_csv("pdf_embeddings.csv", index=False)
            
            st.session_state["df"] = df  # Store in session state

# Chatbot input
st.subheader("💬 Ask ClaudinAi a Question")
query = st.text_input("Type your question based on the PDF content")

if query and "df" in st.session_state:
    df = st.session_state["df"]
    query_embedding = get_embeddings(query)
    df["similarity"] = df["embeddings"].apply(lambda x: cosine_similarity([x], [query_embedding])[0][0])
    
    # Retrieve most relevant content
    top_result = df.sort_values("similarity", ascending=False).iloc[0]["content"]
    
    # Generate response
    prompt = f"Using the following context, answer the question:\n\nContext: {top_result}\n\nQuestion: {query}"
    response = openai.ChatCompletion.create(model=model_choice, messages=[{"role": "user", "content": prompt}])
    
    st.subheader("🧠 ClaudinAi's Answer:")
    st.write(response["choices"][0]["message"]["content"])
