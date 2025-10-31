import streamlit as st
import streamlit.components.v1 as components
import asyncio
import time
from chatbot import RepoChatbot
from main import RepoPipeline           
from knowledge_graph import visualize_repo_graph 
import traceback

# --- Streamlit page setup ---
st.set_page_config(page_title="RepoChat 🤖", page_icon="💬", layout="wide")

st.title("💬 Ask Questions about a GitHub Repository")
st.markdown("Analyze and chat with any public GitHub repo using RAG + Neo4j + PyVis + LLM ⚡")

# --- Session State ---
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "repo_url" not in st.session_state:
    st.session_state.repo_url = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "graph_path" not in st.session_state:
    st.session_state.graph_path = None

# ==========================================================
# 1️⃣ Sidebar: Repository Input
# ==========================================================
st.sidebar.header("⚙️ Load a Repository")
repo_url = st.sidebar.text_input("🔗 GitHub Repository URL", st.session_state.repo_url or "")
analyze_button = st.sidebar.button("🚀 Analyze Repository")

# ==========================================================
# 2️⃣ Full Pipeline: Clone → Analyze → Push Neo4j → Visualize
# ==========================================================
if analyze_button:
    if not repo_url:
        st.sidebar.warning("Please enter a valid GitHub repository URL.")
    else:
        with st.spinner("🔎 Cloning and analyzing repository... this may take a few minutes ⏳"):
            try:
                # --- Step 1: Run full RepoPipeline (Analyzer + Neo4j)
                pipeline = RepoPipeline(repo_url)
                result = asyncio.run(pipeline.run())

                # --- Step 2: Generate PyVis HTML graph
                html_path = visualize_repo_graph()
                st.session_state.graph_path = html_path

                # --- Step 3: Initialize chatbot (RAG-based)
                chatbot = RepoChatbot(repo_url)
                asyncio.run(chatbot.build_and_prepare())

                # --- Save to session
                st.session_state.chatbot = chatbot
                st.session_state.repo_url = repo_url
                st.session_state.chat_history = []

                st.success("✅ Repository analyzed and knowledge graph created successfully!")

            except Exception as e:
                st.error(traceback.format_exc())

# ==========================================================
# 3️⃣ Knowledge Graph Visualization (from Neo4j)
# ==========================================================
st.markdown("---")
st.subheader("🗺️ Knowledge Graph Visualization")

if st.session_state.graph_path:
    with open(st.session_state.graph_path, "r", encoding="utf-8") as f:
        graph_html = f.read()
    components.html(graph_html, height=600, width=950)
else:
    st.info("Run analysis to generate and display the graph.")

# ==========================================================
# 4️⃣ Chat Interface
# ==========================================================
if st.session_state.chatbot:
    chatbot = st.session_state.chatbot
    st.markdown("---")
    st.subheader(f"💬 Chat about `{st.session_state.repo_url}`")

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["q"])
        with st.chat_message("assistant"):
            st.markdown(chat["a"])

    # Chat input
    user_input = st.chat_input("Ask a question about this repository...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("Thinking..."):
                try:
                    answer = asyncio.run(chatbot.chat(user_input))
                except Exception as e:
                    answer = f"⚠️ Error while generating response: {e}"

            # Typing animation
            for chunk in answer.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.03)
            message_placeholder.markdown(full_response)

            # Update session
            st.session_state.chat_history.append({"q": user_input, "a": full_response})
