import os
import json
import asyncio
import tempfile
from git import Repo
from tqdm.asyncio import tqdm_asyncio
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import dotenv

dotenv.load_dotenv()


# ================================================================
# 1️⃣ CLONE GITHUB REPOSITORY
# ================================================================
def clone_repository(repo_url: str) -> str:
    tmp_dir = tempfile.mkdtemp()
    print(f"📦 Cloning {repo_url} into {tmp_dir}...")
    Repo.clone_from(repo_url, tmp_dir)
    print("✅ Repository cloned successfully.")
    return tmp_dir


# ================================================================
# 2️⃣ FILE FILTERS (skip binaries, large files, etc.)
# ================================================================
VALID_EXTENSIONS = [
    ".py", ".js", ".ts", ".tsx", ".md", ".txt", ".json",
    ".yml", ".yaml", ".html", ".css", ".java", ".cpp", ".c", ".sh"
]

def should_summarize(file_path: str) -> bool:
    return any(file_path.endswith(ext) for ext in VALID_EXTENSIONS)


# ================================================================
# 3️⃣ INITIALIZE LLM CHAINS (LangChain)
# ================================================================
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

file_prompt = PromptTemplate(
    input_variables=["filename", "content"],
    template=(
        "You are an expert software engineer. Analyze the code below from the file `{filename}` "
        "and summarize its *functional behavior* in a concise technical explanation.\n\n"
        "Focus on what the code does — not a plain text paraphrase.\n\n"
        "Explain:\n"
        "1. The main purpose of the file.\n"
        "2. Key functions/classes and their roles.\n"
        "3. Important dependencies or integrations.\n"
        "4. How data or control flows through it.\n"
        "5. If applicable, what the file exports or how it's used by other modules.\n\n"
        "Return your answer as a short, structured developer summary (3–6 sentences maximum).\n\n"
        "CODE:\n{content}"
    ),
)
file_chain = LLMChain(llm=llm, prompt=file_prompt)

folder_prompt = PromptTemplate(
    input_variables=["folder_name", "child_summaries"],
    template=(
        "You are analyzing a codebase folder named `{folder_name}`.\n\n"
        "Here are summaries of its contents:\n{child_summaries}\n\n"
        "Combine them into a cohesive technical summary describing:\n"
        "- The overall purpose of this folder.\n"
        "- How its files interact or depend on one another.\n"
        "- What part of a larger application this folder likely represents "
        "(e.g., frontend, backend, utils, data processing, etc.).\n\n"
        "Return your output as a clear, concise developer-oriented summary paragraph."
    ),
)
folder_chain = LLMChain(llm=llm, prompt=folder_prompt)


# ================================================================
# 4️⃣ ASYNC SUMMARIZATION HELPERS
# ================================================================
async def summarize_file(path):
    """Summarize a single file asynchronously."""
    if not should_summarize(path):
        return {"type": "file", "path": path, "summary": "Skipped (unsupported file type)"}

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        content = content[:4000]  # truncate to avoid token overflow
        summary = await file_chain.ainvoke({"filename": os.path.basename(path), "content": content})
        return {"type": "file", "path": path, "summary": summary["text"]}
    except Exception as e:
        return {"type": "file", "path": path, "summary": f"Error: {e}"}


async def summarize_folder(path):
    """Recursively summarize a folder asynchronously."""
    if os.path.isfile(path):
        return await summarize_file(path)

    # Summarize children in parallel
    tasks = []
    for item in os.listdir(path):
        child_path = os.path.join(path, item)
        tasks.append(summarize_folder(child_path))

    children = await tqdm_asyncio.gather(*tasks, desc=f"📁 {os.path.basename(path)}", leave=False)
    child_summaries = "\n".join([f"{os.path.basename(c['path'])}: {c['summary']}" for c in children])

    folder_summary = await folder_chain.ainvoke({
        "folder_name": os.path.basename(path),
        "child_summaries": child_summaries
    })

    return {
        "type": "folder",
        "path": path,
        "summary": folder_summary["text"],
        "children": children
    }


# ================================================================
# 5️⃣ OPTIONAL CACHING (resume partially completed runs)
# ================================================================
CACHE_FILE = "summaries_cache.json"

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_cache(data):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ================================================================
# 6️⃣ MAIN ENTRY
# ================================================================
if __name__ == "__main__":
    repo_url = "https://github.com/Ebrukoksal/quizzy"

    repo_dir = clone_repository(repo_url)

    print("⚙️ Starting summarization...")
    result = asyncio.run(summarize_folder(repo_dir))

    # Save final results
    with open("summaries.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("✅ Summaries saved to summaries.json")
