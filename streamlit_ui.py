import streamlit as st
import asyncio
import time
from chatbot import build_rag_chain, chat_with_repo

st.set_page_config(page_title="RepoChat 🤖", page_icon="💬", layout="wide")

st.title("💬 Ask Questions about a GitHub Repository")
st.markdown("Analyze and chat with any public GitHub repo using RAG + Qdrant + LLM.")

# --- Session State ---
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "repo_url" not in st.session_state:
    st.session_state.repo_url = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: Load Repository ---
st.sidebar.header("⚙️ Load a Repository")
repo_url = st.sidebar.text_input("🔗 GitHub Repository URL", st.session_state.repo_url or "")
analyze_button = st.sidebar.button("🚀 Analyze Repository")

# ==========================================================
# 1️⃣ Build RAG Chain
# ==========================================================
if analyze_button:
    if not repo_url:
        st.sidebar.warning("Please enter a valid GitHub repository URL.")
    else:
        with st.spinner("Cloning and analyzing repository... this may take a few minutes ⏳"):
            try:
                rag_chain, _ = asyncio.run(build_rag_chain(repo_url))
                st.session_state.rag_chain = rag_chain
                st.session_state.repo_url = repo_url
                st.session_state.chat_history = []
                st.success("✅ Repository analyzed successfully! You can now chat below.")
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")

# ==========================================================
# 2️⃣ Chat Interface
# ==========================================================
if st.session_state.rag_chain:
    st.markdown("---")
    st.subheader(f"💬 Chat about `{st.session_state.repo_url}`")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["q"])
        with st.chat_message("assistant"):
            st.markdown(chat["a"])

    # User input
    user_input = st.chat_input("Ask a question about this repository...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                try:
                    answer = asyncio.run(chat_with_repo(st.session_state.rag_chain, user_input))
                except Exception as e:
                    answer = f"⚠️ Error while generating response: {e}"

            # --- Typing effect ---
            for chunk in answer.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.03)  # controls typing speed

            # finalize output
            message_placeholder.markdown(full_response)
            st.session_state.chat_history.append({"q": user_input, "a": full_response})
