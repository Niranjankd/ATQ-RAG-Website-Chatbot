from dotenv import load_dotenv
import os
from groq import Groq
load_dotenv()

import streamlit as st
import requests
from bs4 import BeautifulSoup

import chromadb
from sentence_transformers import SentenceTransformer

def split_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    
    for i in range(0,len(words), chunk_size):
        chunk= " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

model = SentenceTransformer("all-MiniLM-L6-v2")

client_groq = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="website_data"
)

st.title("🤖 ATQ - Any Time Questions")

st.subheader(
    "Ask Any Website. Get Answers Anytime."
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
with st.sidebar:
    st.header("Chat History")
    
    for chat in st.session_state.chat_history:
        st.write("Q:", chat["question"])
        st.write("A:", chat["answer"])
        st.divider()

website_url = st.text_input(
    "Enter Website URL"
)    
    
question = st.text_input(
    "ASK a Question",
    value=st.session_state.get("selected_question", "")
)

if st.button("ASK"):

    if question:

        results = collection.query(
            query_embeddings=[
                model.encode(question).tolist()
            ],
            n_results=1
        )

        context = results["documents"][0][0]

        prompt = f"""
        Answer the question using only the website content below.

        Website Content:
        {context}

        Question:
        {question}
        """

        response = client_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content
        
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            st.write(answer)
        
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
        
        with st.expander("Source Context"):
            st.write(context)

if st.button("Index Website"):

    if website_url:

        try:
            response = requests.get(website_url)

            if response.status_code == 200:

                soup = BeautifulSoup(
                    response.text,
                    "html.parser"
                )

                content = soup.get_text()
                
                st.session_state.website_content = content
                st.session_state.indexed_url = website_url
                
                chunks = split_text(content)
                
                word_count = len(content.split())
                chunk_count = len(chunks)
                
                st.subheader("Website Dashboard")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Words", word_count)

                with col2:
                    st.metric("Chunks", chunk_count)

                with col3:
                    st.metric("URL", "1")
                    
                st.info(f"Indexed URL: {website_url}")
                
                st.write(f"Created {len(chunks)} chunks")
                # st.write("Embedding started...")
                
                embeddings = model.encode(chunks).tolist()
                
                # st.write("Embedding completed...")
                
                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=[str(i) for i in range(len(chunks))]
                )
                st.success("Chunks stored in ChromaDB!")
                
                st.success("Website scraped successfully!")
                
                st.subheader("Suggested Questions")
                                    
                st.text_area(
                    "Scraped Content",
                    content[:5000],
                    height=300
                )

            else:
                st.error("Unable to access website")

        except Exception as e:
            st.error(f"Error: {e}")

    else:
        st.warning(
            "Please enter a website URL"
        )
        
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

suggested_questions = [
    "What is this website about?",
    "What services are offered?",
    "What are the main features?"
]

for q in suggested_questions:
    if st.button(q):
        st.session_state.selected_question = q
        st.rerun()
        
if st.button("Summarize Website"):
    if "website_content" not in st.session_state:
        st.warning("Please index a website first.")
        st.stop()
        
    summary_prompt = f"""
    Summarize the website content below in simple points.

    Website Content:
    {st.session_state.website_content[:4000]}
    """

    summary_response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": summary_prompt
            }
        ]
    )

    st.subheader("Website Summary")
    st.write(summary_response.choices[0].message.content)