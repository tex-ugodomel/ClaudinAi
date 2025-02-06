import streamlit as st
import openai
import numpy as np
import pandas as pd
import PyPDF2
import io
import tiktoken
from typing import List, Dict
import time

# Page configuration
st.set_page_config(page_title="ClaudinAI", page_icon="🤖", layout="wide")

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings_db" not in st.session_state:
    st.session_state.embeddings_db = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        # Fallback: rough estimate if tiktoken fails
        return len(text.split()) * 1.3

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        if end < text_length:
            last_period = max(
                text.rfind('.', start, end),
                text.rfind('\n', start, end)
            )
            if last_period > start:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    
    return chunks

def read_pdf(file) -> str:
    try:
        file_bytes = file.getvalue()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def read_txt(file) -> str:
    try:
        return file.getvalue().decode('utf-8')
    except Exception as e:
        st.error(f"Error reading TXT file: {str(e)}")
        return ""

def get_embedding(text: str, client) -> List[float]:
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None

def process_document(file_content: str, progress_bar, client) -> pd.DataFrame:
    if not file_content:
        return None
    
    chunks = chunk_text(file_content)
    total_chunks = len(chunks)
    
    if not chunks:
        st.error("No text content found in document")
        return None
    
    embeddings = []
    texts = []
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk, client)
            if embedding:
                embeddings.append(embedding)
                texts.append(chunk)
            progress_bar.progress((i + 1) / total_chunks)
        except Exception as e:
            st.error(f"Error processing chunk {i+1}: {str(e)}")
            continue
    
    if not embeddings:
        return None
        
    return pd.DataFrame({
        'text': texts,
        'embedding': embeddings
    })

def get_context(query: str, df: pd.DataFrame, client, top_n: int = 3) -> str:
    try:
        query_embedding = get_embedding(query, client)
        if not query_embedding:
            return ""
        
        similarities = df.embedding.apply(lambda x: np.dot(x, query_embedding))
        top_contexts = df.assign(similarity=similarities)\
                        .nlargest(top_n, 'similarity')\
                        .apply(lambda x: f"{x['text']}", axis=1)\
                        .tolist()
        
        return " ".join(top_contexts)
    except Exception as e:
        st.error(f"Error retrieving context: {str(e)}")
        return ""

# Sidebar
with st.sidebar:
    st.title("ClaudinAI Settings")
    api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    st.markdown("### Knowledge Base Settings")
    uploaded_file = st.file_uploader("Upload Knowledge Base (PDF, TXT)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            st.session_state.uploaded_file_content = read_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            st.session_state.uploaded_file_content = read_txt(uploaded_file)
    
    if st.session_state.uploaded_file_content and api_key:
        st.markdown("### Processing Settings")
        chunk_size = st.slider("Chunk Size (characters)", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 50, 200, 100)
        
        if st.button("Process Document"):
            try:
                client = openai.OpenAI(api_key=api_key)
                progress_bar = st.progress(0)
                st.session_state.embeddings_db = process_document(
                    st.session_state.uploaded_file_content,
                    progress_bar,
                    client
                )
                if st.session_state.embeddings_db is not None:
                    st.session_state.processing_complete = True
                    st.success("Document processed successfully!")
                progress_bar.empty()
            except Exception as e:
                st.error(f"Error during document processing: {str(e)}")

# Main chat interface
st.title("ClaudinAI")
st.markdown("Your intelligent chatbot powered by GPT-4o-mini 🚀")

# Memory management
max_messages = 10
if len(st.session_state.messages) > max_messages:
    st.session_state.messages = st.session_state.messages[-max_messages:]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and processing
if prompt := st.chat_input("What's on your mind?"):
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    else:
        try:
            client = openai.OpenAI(api_key=api_key)
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get context if available
            context = ""
            if st.session_state.embeddings_db is not None:
                with st.spinner("Retrieving relevant context..."):
                    context = get_context(prompt, st.session_state.embeddings_db, client)
            
            # Prepare messages
            system_message = (
                "You are ClaudinAI, a helpful and knowledgeable assistant. "
                + ("Using the following context to inform your response: " + context if context else "")
            )
            
            messages = [{"role": "system", "content": system_message}] + st.session_state.messages
            
            # Token counting and management
            total_tokens = sum(count_tokens(msg["content"]) for msg in messages)
            if total_tokens > 15000:  # Safety limit
                messages = messages[-5:]  # Keep only recent messages
            
            # Get response
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-4o-mini",  # Updated model name
                    messages=messages,
                    temperature=0.7,
                    max_tokens=2000,
                )
            
            # Display response
            assistant_response = response.choices[0].message.content
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if "Rate limit" in str(e):
                time.sleep(1)

# Enhanced CSS styling
st.markdown("""
    <style>
    .stChat {
        border-radius: 10px;
        padding: 10px;
        background-color: #f0f2f6;
    }
    .stChatMessage {
        padding: 10px;
        margin: 5px 0;
        border-radius: 15px;
    }
    .stChatInput {
        border-radius: 20px;
        border: 2px solid #4CAF50;
    }
    .stButton button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
