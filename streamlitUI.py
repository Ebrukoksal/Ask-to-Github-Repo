import streamlit as st
import time
import asyncio
from create_summaries import clone_repository, summarize_folder

st.title("Ask To Github Repo")
repo_url = st.text_input("Enter the Github Repo URL")
if st.button("Load"):
    with st.spinner("Thinking..."):
        time.sleep(2)
        repo_dir = clone_repository(repo_url)
        result = asyncio.run(summarize_folder(repo_dir))
        st.write(result)