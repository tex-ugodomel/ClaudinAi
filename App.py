import streamlit as st
import openai
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

# Sidebar for API Key input
st.sidebar.header("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

# Store API key in session state
if api_key:
    st.session_state["OPENAI_API_KEY"] = api_key

# Ensure API key is set before making OpenAI API calls
def get_openai_api_key():
    return st.session_state.get("OPENAI_API_KEY", None)

# Load preprocessed document embeddings
DATA_PATH = "data/parsed_pdf_docs_with_embeddings.csv"
df = pd.read_csv(DATA_PATH)
df["embeddings"] = df.embeddings.apply(json.loads).apply(np.array)  # Convert stored embeddings back to arrays

# Function to get embeddings using OpenAI API
def get_embeddings(text):
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

# Search function to find the most relevant content
def search_content(input_text, top_k=3):
    embedded_value = get_embeddings(input_text)
    if embedded_value is None:
        return None
    
    df["similarity"] = df.embeddings.apply(lambda x: cosine_similarity(x.reshape(1, -1), embedded_value.reshape(1, -1))[0][0])
    return df.sort_values("similarity", ascending=False).head(top_k)

# Generate response using retrieved documents
def generate_output(input_prompt, similar_content, threshold=0.5):
    api_key = get_openai_api_key()
    if not api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
        return None

    content = similar_content.iloc[0]["content"] if not similar_content.empty else ""

    for i, row in similar_content.iterrows():
        if row["similarity"] > threshold:
            content += f"\n\n{row['content']}"

    prompt = f"INPUT PROMPT:\n{input_prompt}\n-------\nCONTENT:\n{content}"

    system_prompt = '''
        You will be provided with an input prompt and content as context that can be used to reply to the prompt.
        
        1. First, assess whether the provided content is relevant to the input prompt.  
        2a. If relevant, use the content to generate a response.  
        2b. If irrelevant, reply using general knowledge or state that the answer is unknown.  
        
        Stay concise, using only the necessary content for the response.
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

# Streamlit UI
st.title("RAG Chatbot")

user_input = st.text_input("Ask a question:")

if user_input:
    matching_content = search_content(user_input, top_k=3)

    if matching_content is not None:
        st.write("### Retrieved Context")
        for i, row in matching_content.iterrows():
            st.write(f"**Match {i+1} (Similarity: {row['similarity']:.2f})**")
            st.write(row["content"][:300] + "...")  # Show preview of retrieved content

        st.write("### Response")
        response = generate_output(user_input, matching_content)
        if response:
            st.write(response)
