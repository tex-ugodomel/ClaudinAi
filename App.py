import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import openai
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------- APP CONFIGURATION ---------------------- #
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# File paths
DATA_PATH = "data/parsed_pdf_docs_with_embeddings.csv"
UPLOAD_FOLDER = "uploaded_docs"

# Ensure required directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Clear or create the embeddings file on startup
if not os.path.exists(DATA_PATH) or os.stat(DATA_PATH).st_size == 0:
    pd.DataFrame(columns=["content", "embeddings"]).to_csv(DATA_PATH, index=False)

# ---------------------- SIDEBAR SETTINGS ---------------------- #
st.sidebar.header("Settings")

# API Key Input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

# ---------------------- UTILITY FUNCTIONS ---------------------- #

def get_openai_api_key():
    """Retrieve OpenAI API key from session state."""
    return st.session_state.get("OPENAI_API_KEY", None)

def get_embeddings(text):
    """Fetch embeddings from OpenAI API."""
    api_key = get_openai_api_key()
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None
    
    try:
        client = openai.OpenAI(api_key=api_key)
        embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float"
        )
        return np.array(embeddings.data[0].embedding)
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

def search_content(input_text, top_k=3):
    """Retrieve the most relevant chunks based on cosine similarity."""
    df = pd.read_csv(DATA_PATH)
    
    if df.empty or "embeddings" not in df.columns:
        st.warning("No documents available. Please upload files first.")
        return None

    df["embeddings"] = df.embeddings.apply(json.loads).apply(np.array)
    query_embedding = get_embeddings(input_text)

    if query_embedding is None:
        return None

    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x.reshape(1, -1), query_embedding.reshape(1, -1))[0][0])
    return df.sort_values("similarity", ascending=False).head(top_k)

def generate_output(input_prompt, similar_content, threshold=0.5):
    """Generate a response using retrieved context."""
    api_key = get_openai_api_key()
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None

    if similar_content is None or similar_content.empty:
        return "I couldn't find relevant information in the documents."

    content = similar_content.iloc[0]["content"]
    for _, row in similar_content.iterrows():
        if row["similarity"] > threshold:
            content += f"\n\n{row['content']}"

    prompt = f"INPUT PROMPT:\n{input_prompt}\n-------\nCONTENT:\n{content}"
    
    system_prompt = '''
        You will receive an input prompt and relevant content as context.
        1. If the content is relevant, generate a response using it.
        2. If not relevant, use general knowledge or state that you don't know.
        Keep responses concise and avoid unnecessary information.
    '''

    try:
        client = openai.OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.5,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# ---------------------- FILE UPLOAD SECTION ---------------------- #
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDFs or text files", accept_multiple_files=True, type=["txt", "pdf"])

def extract_text_from_file(file_path):
    """Extract text from a file (PDF or TXT)."""
    if file_path.endswith(".pdf"):
        return extract_text(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

if uploaded_files:
    st.sidebar.write("Processing uploaded files...")
    contents = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = extract_text_from_file(file_path)
        if text.strip():
            contents.append({"content": text, "embeddings": get_embeddings(text)})
    
    if contents:
        df = pd.DataFrame(contents)
        df.to_csv(DATA_PATH, index=False)  # Overwrite embeddings file with new data
        st.sidebar.success("Files processed and indexed!")
    else:
        st.sidebar.error("No text could be extracted. Ensure your files contain readable text.")

# ---------------------- CHATBOT UI ---------------------- #
st.title("📚 Chat with Your Documents")

# Chat display
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Bottom Input Bar
user_input = st.chat_input("Ask a question about your uploaded documents...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process query
    retrieved_content = search_content(user_input, top_k=3)
    response = generate_output(user_input, retrieved_content)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
