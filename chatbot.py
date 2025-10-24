import os
import asyncio
import tempfile
from git import Repo
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from helper import RepositoryAnalyzer 

load_dotenv()


class RepoChatbot:

    def __init__(
        self,
        repo_url: str,
        qdrant_location=":memory:",
        model="gpt-4.1-nano",
        embedding_model="text-embedding-3-small",
        temperature=0,
        k=5,
    ):
        self.repo_url = repo_url
        self.qdrant_location = qdrant_location
        self.embedding_model = embedding_model
        self.llm_model = model
        self.temperature = temperature
        self.k = k

        self.repo_dir = None
        self.analysis = None
        self.qdrant = None
        self.retriever = None
        self.rag_chain = None
        self.analyzer = RepositoryAnalyzer(self.repo_url)

    # ================================================================
    # 1️⃣ CLONE GITHUB REPOSITORY
    # ================================================================
    def clone_repository(self) -> str:
        tmp_dir = tempfile.mkdtemp()
        print(f"📦 Cloning {self.repo_url} into {tmp_dir}...")
        Repo.clone_from(self.repo_url, tmp_dir)
        print("✅ Repository cloned successfully.")
        self.repo_dir = tmp_dir
        return tmp_dir

    # ================================================================
    # 2️⃣ ANALYZE REPOSITORY
    # ================================================================
    async def analyze_repository(self):
        if not self.repo_dir:
            self.clone_repository()
        print("🔍 Analyzing repository...")
        self.analysis = await self.analyzer.analyze_folder(self.repo_dir)
        print("✅ Analysis complete.")
        return self.analysis

    # ================================================================
    # 3️⃣ FLATTEN REPO SUMMARIES FOR EMBEDDING
    # ================================================================
    def _flatten_analysis(self, node):
        results = []
        if "file_path" in node:
            text = f"{node['file_path']}\n{node.get('goal', '')}\n{node.get('summary', '')}"
            results.append(text)
        elif "children" in node:
            for c in node["children"]:
                results.extend(self._flatten_analysis(c))
        return results

    # ================================================================
    # 4️⃣ BUILD QDRANT VECTOR STORE
    # ================================================================
    def build_vector_store(self):
        if not self.analysis:
            raise ValueError("Analysis not found. Run analyze_repository() first.")

        texts = self._flatten_analysis(self.analysis)
        print(f"🧠 Indexing {len(texts)} documents into Qdrant...")

        embeddings = OpenAIEmbeddings(model=self.embedding_model)
        self.qdrant = Qdrant.from_texts(
            texts,
            embedding=embeddings,
            location=self.qdrant_location,
            collection_name="repo_knowledge"
        )

        self.retriever = self.qdrant.as_retriever(search_kwargs={"k": self.k})
        print("✅ Vector store built successfully.")

    # ================================================================
    # 5️⃣ BUILD RAG CHAIN
    # ================================================================
    def build_rag_chain(self):
        if not self.retriever:
            raise ValueError("Retriever not initialized. Run build_vector_store() first.")

        llm = ChatOpenAI(model=self.llm_model, temperature=self.temperature)
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA

        CUSTOM_PROMPT = PromptTemplate(
            template=(
                "You are an expert software engineer analyzing a GitHub repository.\n"
                "Use the provided repository context to answer the user's question clearly and concisely.\n\n"
                "Repository Context:\n{context}\n\n"
                "User Question: {question}\n\n"
                "Your answer:"
            ),
            input_variables=["context", "question"],
        )

        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.retriever,
            chain_type_kwargs={"prompt": CUSTOM_PROMPT},
        )

        print("🤖 RAG chain created with custom prompt and ready for chatting.")
        return self.rag_chain

    # ================================================================
    # 6️⃣ CHAT INTERFACE
    # ================================================================
    async def chat(self, query: str) -> str:
        """Run a chat query through the RAG chain asynchronously."""
        if not self.rag_chain:
            raise ValueError("RAG chain not built. Run build_rag_chain() first.")
        print(f"💬 Query: {query}")
        return self.rag_chain.run(query)

    # ================================================================
    # 7️⃣ FULL PIPELINE (Convenience Method)
    # ================================================================
    async def build_and_prepare(self):
        """One-shot pipeline to clone, analyze, and build RAG chain."""
        self.clone_repository()
        await self.analyze_repository()
        self.build_vector_store()
        self.build_rag_chain()
        print("✅ RepoChatbot is ready for questions!")
        return self


# ================================================================
# 8️⃣ Example usage
# ================================================================
if __name__ == "__main__":
    async def main():
        repo_url = "https://github.com/Ebrukoksal/quizzy"
        chatbot = RepoChatbot(repo_url)
        await chatbot.build_and_prepare()

        response = await chatbot.chat("What is the main purpose of this repository?")
        print("\n💡 Assistant:", response)

    asyncio.run(main())
