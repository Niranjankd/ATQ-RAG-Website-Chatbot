import streamlit as st
import requests
from bs4 import BeautifulSoup

st.title("🤖 ATQ - Any Time Questions")

st.subheader(
    "Ask Any Website. Get Answers Anytime."
)

website_url = st.text_input(
    "Enter Website URL"
)

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

                st.success("Website scraped successfully!")

                st.text_area(
                    "Scraped Content",
                    content[:5000],
                    height=300
                )

            else:
                st.error("Unable to access website")

        except Exception:
            st.error("Invalid URL")

    else:
        st.warning(
            "Please enter a website URL"
        )