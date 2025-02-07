import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import openai
from pdfminer.high_level import extract_text
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_path
import base64
import io

# ---------------------- APP CONFIGURATION ---------------------- #
st.set_page_config(page_title="ClaudinAi Chatbot", layout="wide")

# File paths
DATA_PATH = "data/parsed_pdf_docs_with_embeddings.csv"
IMAGE_DATA_PATH = "data/parsed_pdf_image_docs.json"
UPLOAD_FOLDER = "uploaded_docs"

# Ensure required directories exist
os.makedirs("data", exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------------- UTILITY FUNCTIONS ---------------------- #

def get_img_uri(img):
    """Convert image to base64 encoded data URI."""
    png_buffer = io.BytesIO()
    img.save(png_buffer, format="PNG")
    png_buffer.seek(0)
    base64_png = base64.b64encode(png_buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{base64_png}"

def analyze_pdf_images(file_path):
    """Analyze PDF pages using GPT-4o."""
    api_key = st.session_state.get("OPENAI_API_KEY")
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return []

    client = openai.OpenAI(api_key=api_key)
    
    # Convert PDF to images
    images = convert_from_path(file_path)
    
    # System prompt for image analysis
    system_prompt = '''
    You will be provided with an image of a PDF page or a slide. Your goal is to deliver a detailed and engaging description.
    - Describe visual elements in detail
    - Focus on the content itself
    - Explain technical terms in simple language
    - Provide comprehensive yet concise explanation
    '''
    
    page_descriptions = []
    
    # Analyze each image (skip first page if it's an intro)
    for img in images[1:]:
        img_uri = get_img_uri(img)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": img_uri}
                            }
                        ]
                    },
                ],
                max_tokens=500,
                temperature=0,
                top_p=0.1
            )
            page_descriptions.append(response.choices[0].message.content)
        except Exception as e:
            st.error(f"Error analyzing image: {e}")
    
    return page_descriptions

def get_embeddings(text, api_key):
    """Fetch embeddings from OpenAI API."""
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

def search_content(input_text, method='text', top_k=3):
    """Retrieve the most relevant chunks based on cosine similarity."""
    if method == 'text':
        # Text-based search
        df = pd.read_csv(DATA_PATH)
        
        if df.empty or "embeddings" not in df.columns:
            st.warning("No text documents available. Please upload text files first.")
            return None

        df["embeddings"] = df.embeddings.apply(json.loads).apply(np.array)
        query_embedding = get_embeddings(input_text, st.session_state.get("OPENAI_API_KEY"))

        if query_embedding is None:
            return None

        df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x.reshape(1, -1), query_embedding.reshape(1, -1))[0][0])
        return df.sort_values("similarity", ascending=False).head(top_k)
    
    elif method == 'image':
        # Image-based search
        with open(IMAGE_DATA_PATH, 'r') as f:
            image_docs = json.load(f)
        
        # Simple text-based search on image descriptions
        matching_docs = []
        for doc in image_docs:
            for page_desc in doc['pages_description']:
                similarity = cosine_similarity(
                    get_embeddings(page_desc, st.session_state.get("OPENAI_API_KEY")).reshape(1, -1),
                    get_embeddings(input_text, st.session_state.get("OPENAI_API_KEY")).reshape(1, -1)
                )[0][0]
                matching_docs.append({
                    'content': page_desc,
                    'similarity': similarity
                })
        
        # Sort and return top k
        matching_docs.sort(key=lambda x: x['similarity'], reverse=True)
        return pd.DataFrame(matching_docs[:top_k])

def generate_output(input_prompt, similar_content, method='text', threshold=0.5):
    """Generate a response using retrieved context."""
    api_key = st.session_state.get("OPENAI_API_KEY")
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

# ---------------------- SIDEBAR SETTINGS ---------------------- #
st.sidebar.header("Settings")

# API Key Input
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

# RAG Method Selection
rag_method = st.sidebar.selectbox(
    "Select RAG Method", 
    ["Text Embedding", "Image Analysis"]
)

# ---------------------- FILE UPLOAD SECTION ---------------------- #
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs or text files", 
    accept_multiple_files=True, 
    type=["txt", "pdf", "py"]
)

def process_uploaded_files(uploaded_files, method):
    """Process uploaded files based on the selected method."""
    if not uploaded_files:
        return False

    st.sidebar.write("Processing uploaded files...")
    
    if method == "Text Embedding":
        contents = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract text based on file type
            if file_path.endswith(".pdf"):
                text = extract_text(file_path)
            elif file_path.endswith((".txt", ".py")):
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                continue

            if text.strip():
                embeddings = get_embeddings(text, st.session_state.get("OPENAI_API_KEY"))
                if embeddings is not None:
                    contents.append({"content": text, "embeddings": embeddings.tolist()})
        
        if contents:
            df = pd.DataFrame(contents)
            df.to_csv(DATA_PATH, index=False)
            st.sidebar.success("Text files processed and indexed!")
            return True

    elif method == "Image Analysis":
        image_docs = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".pdf"):
                file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Analyze PDF images
                pages_description = analyze_pdf_images(file_path)
                
                image_docs.append({
                    "filename": uploaded_file.name,
                    "pages_description": pages_description
                })
        
        if image_docs:
            with open(IMAGE_DATA_PATH, 'w') as f:
                json.dump(image_docs, f)
            st.sidebar.success("PDF images processed and indexed!")
            return True

    st.sidebar.error("No processable content found in the uploaded files.")
    return False

if uploaded_files:
    process_uploaded_files(uploaded_files, rag_method)

# ---------------------- CHATBOT UI ---------------------- #
st.title("📚 ClaudinAi Chatbot")

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
    method = 'text' if rag_method == "Text Embedding" else 'image'
    retrieved_content = search_content(user_input, method=method, top_k=3)
    response = generate_output(user_input, retrieved_content, method=method)

    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)
