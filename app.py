import streamlit as st

st.set_page_config(
    page_title="ATQ - Any Time Questions",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ATQ - Any Time Questions")

st.subheader(
    "Ask Any Website. Get Answers Anytime."
)

website_url = st.text_input(
    "Enter Website URL"
)

if st.button("Index Website"):
    if website_url:
        st.success(
            f"Website entered: {website_url}"
        )
    else:
        st.warning(
            "Please enter a website URL"
        )