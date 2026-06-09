from dotenv import load_dotenv
import os
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

import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="website_data"
)

st.title("🤖 ATQ - Any Time Questions")

st.subheader(
    "Ask Any Website. Get Answers Anytime."
)

website_url = st.text_input(
    "Enter Website URL"
)

question = st.text_input(
    "ASK a Question"
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

        st.subheader("Answer")
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
                
                chunks = split_text(content)
                
                st.write(f"Created {len(chunks)} chunks")
                st.write("Embedding started...")
                
                embeddings = model.encode(chunks).tolist()
                
                st.write("Embedding completed...")
                
                collection.add(
                    documents=chunks,
                    embeddings=embeddings,
                    ids=[str(i) for i in range(len(chunks))]
                )
                st.success("Chunks stored in ChromaDB!")
                
                st.success("Website scraped successfully!")

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
        