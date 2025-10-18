import os
import re
import ast
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
# 2️⃣ DETECT FILE LANGUAGE
# ================================================================
LANGUAGE_MAP = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".tsx": "React (TypeScript)", ".jsx": "React (JavaScript)",
    ".java": "Java", ".cpp": "C++", ".c": "C",
    ".html": "HTML", ".css": "CSS", ".json": "JSON",
    ".yml": "YAML", ".yaml": "YAML", ".md": "Markdown",
    ".sh": "Shell"
}

def detect_language(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    return LANGUAGE_MAP.get(ext, "Unknown")


# ================================================================
# 3️⃣ CODE ATTRIBUTE EXTRACTORS
# ================================================================
def extract_python_attributes(code: str):
    """Extract functions, classes, and imports from Python code."""
    try:
        tree = ast.parse(code)
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        imports = [
            n.names[0].name if hasattr(n, "names") else ""
            for n in ast.walk(tree)
            if isinstance(n, (ast.Import, ast.ImportFrom))
        ]
        return functions, classes, imports
    except Exception:
        return [], [], []


def extract_basic_attributes(code: str, language: str):
    """Fallback regex-based function/class extraction for non-Python files."""
    func_pattern = r"(?:def|function|fn|void|int|float|const|let)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    class_pattern = r"(?:class)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
    import_pattern = r"(?:import|from)\s+([a-zA-Z0-9_\.\/\-]+)"

    functions = re.findall(func_pattern, code)
    classes = re.findall(class_pattern, code)
    imports = re.findall(import_pattern, code)
    return functions, classes, imports


# ================================================================
# 4️⃣ LLM SETUP
# ================================================================
llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

goal_prompt = PromptTemplate(
    input_variables=["filename", "content"],
    template=(
        "You are an expert software engineer.\n"
        "In one clear sentence, describe the **goal** of this code file `{filename}`.\n"
        "Focus on what this code is designed to achieve or its role in the system.\n\n"
        "CODE:\n{content}"
    ),
)
summary_prompt = PromptTemplate(
    input_variables=["filename", "content"],
    template=(
        "You are an expert developer writing technical documentation.\n"
        "Summarize `{filename}` in 3–5 sentences explaining **how** the code works, "
        "its structure, and interactions between components.\n\n"
        "CODE:\n{content}"
    ),
)

goal_chain = LLMChain(llm=llm, prompt=goal_prompt)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)


# ================================================================
# 5️⃣ ASYNC SUMMARIZER
# ================================================================
async def analyze_file(path):
    language = detect_language(path)

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        return {"file_path": path, "error": str(e)}

    # Extract static attributes
    if language == "Python":
        functions, classes, imports = extract_python_attributes(content)
    else:
        functions, classes, imports = extract_basic_attributes(content, language)

    # Truncate for context
    truncated = content[:4000]

    # Generate goal and summary concurrently
    goal_task = goal_chain.ainvoke({"filename": os.path.basename(path), "content": truncated})
    summary_task = summary_chain.ainvoke({"filename": os.path.basename(path), "content": truncated})
    goal, summary = await asyncio.gather(goal_task, summary_task)

    return {
        "file_path": path,
        "language": language,
        "functions": functions,
        "classes": classes,
        "dependencies": imports,
        "goal": goal["text"],
        "summary": summary["text"],
        "file_content": content[:2000]  # keep only a small snippet
    }


async def analyze_folder(path):
    if os.path.isfile(path):
        return await analyze_file(path)

    tasks = []
    for item in os.listdir(path):
        child_path = os.path.join(path, item)
        tasks.append(analyze_folder(child_path))

    children = await tqdm_asyncio.gather(*tasks, desc=f"📁 {os.path.basename(path)}", leave=False)
    return {"folder_path": path, "children": children}


# ================================================================
# 6️⃣ MAIN
# ================================================================
if __name__ == "__main__":
    repo_url = "https://github.com/Ebrukoksal/quizzy"
    repo_dir = clone_repository(repo_url)

    print("⚙️ Analyzing repository structure and attributes...")
    result = asyncio.run(analyze_folder(repo_dir))

    with open("repo_attributes.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("✅ Attributes extracted and saved to repo_attributes.json")
