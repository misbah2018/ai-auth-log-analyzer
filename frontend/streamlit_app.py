
import streamlit as st
import pandas as pd
import requests

st.title("AI Auth Log Analyzer")

uploaded_file = st.file_uploader("Upload your Linux auth.log file", type="log")

if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    st.text_area("Log Preview", content[:1000], height=300)

    if st.button("Analyze Log"):
        with open("temp_auth.log", "w") as temp_file:
            temp_file.write(content)
        files = {"file": open("temp_auth.log", "rb")}
        response = requests.post("http://127.0.0.1:8000/analyze", files=files)

        if response.status_code == 200:
            result = response.json()
            st.success("Log Analysis Complete")
            st.json(result)
        else:
            st.error("Failed to analyze log")
    