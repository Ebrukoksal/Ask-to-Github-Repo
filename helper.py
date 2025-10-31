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
from langchain.schema.output_parser import StrOutputParser
import dotenv
import stat
import shutil


class RepositoryAnalyzer:
    """
    A unified class for:
      - Cloning GitHub repositories
      - Detecting file languages
      - Extracting code attributes
      - Generating LLM-based goals and summaries
    """

    def __init__(self, repo_url: str, language_map_path="language_map.json", prompts_path="prompts.json"):
        dotenv.load_dotenv()

        self.repo_url = repo_url
        self.repo_dir = None

        # Load language mappings and prompts
        with open(language_map_path, "r", encoding="utf-8") as f:
            self.language_map = json.load(f)

        with open(prompts_path, "r", encoding="utf-8") as f:
            self.prompts = json.load(f)

        # Initialize LLM and pipelines
        self.llm = ChatOpenAI(
            model="gpt-4.1-nano",
            temperature=0,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        goal_prompt = PromptTemplate(
            input_variables=self.prompts["goal_prompt"]["input_variables"],
            template=self.prompts["goal_prompt"]["template"]
        )
        summary_prompt = PromptTemplate(
            input_variables=self.prompts["summary_prompt"]["input_variables"],
            template=self.prompts["summary_prompt"]["template"]
        )

        self.goal_chain = goal_prompt | self.llm | StrOutputParser()
        self.summary_chain = summary_prompt | self.llm | StrOutputParser()

    # ================================================================
    # 1️⃣ Clone GitHub Repository
    # ================================================================
    def clone_repository(self) -> str:
        import shutil
        import pathlib
        import os
        from git import Repo

        base_dir = os.path.join(os.getcwd(), "repos")
        os.makedirs(base_dir, exist_ok=True)

        repo_name = self.repo_url.rstrip("/").split("/")[-1]
        repo_path = os.path.join(base_dir, repo_name)

        if os.path.exists(repo_path):
            print(f"⚠️  {repo_path} already exists. Removing old copy...")
            self.safe_rmtree(repo_path)
            if os.path.exists(repo_path):
                print("IT EXISTS")
 
        print(f"📦 Cloning {self.repo_url} into {repo_path}...")
        Repo.clone_from(self.repo_url, repo_path)
        self.fix_permissions_after_clone(repo_path)
        print("✅ Repository cloned successfully.")

        self.repo_dir = str(pathlib.Path(repo_path).resolve())
        return self.repo_dir

    def fix_permissions_after_clone(self, folder_path):
        """Fix permissions on a freshly cloned git repository"""
        if not os.path.exists(folder_path):
            return
        
        # Set proper permissions for directories and files
        for root, dirs, files in os.walk(folder_path):
            # Fix directory permissions
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    # Give full permissions to directories
                    os.chmod(dir_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                except Exception as e:
                    print(f"Could not fix permissions for directory {dir_path}: {e}")
            
            # Fix file permissions
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    # Give read/write permissions to files
                    os.chmod(file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
                except Exception as e:
                    print(f"Could not fix permissions for file {file_path}: {e}")
    def safe_rmtree(self, folder_path):
        """
        Safely remove a directory tree
        """
        def handle_remove_readonly(func, path, exc):
            """Fix permissions and retry deletion"""
            if os.path.exists(path):
                os.chmod(path, stat.S_IWRITE)
                func(path)
        
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path, onerror=handle_remove_readonly)
            return True
        return False

    # ================================================================
    # 2️⃣ Detect File Language
    # ================================================================
    def detect_language(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        return self.language_map.get(ext, "Unknown")

    # ================================================================
    # 3️⃣ Code Attribute Extraction
    # ================================================================
    def extract_code_attributes(self, code: str, language: str):
        """Unified extractor for functions, classes, and imports."""

        def _extract_python(code):
            functions, classes, imports = [], [], []
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                    elif isinstance(node, (ast.Import, ast.ImportFrom)):
                        for alias in getattr(node, "names", []):
                            imports.append(alias.name)
            except SyntaxError:
                func_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)"
                class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)"
                import_pattern = r"(?:import|from)\s+([a-zA-Z0-9_\.]+)"
                functions = re.findall(func_pattern, code)
                classes = re.findall(class_pattern, code)
                imports = re.findall(import_pattern, code)
            return functions, classes, imports

        def _extract_js_like(code):
            func_pattern = r"(?:function\s+|const\s+|let\s+|async\s+function\s+|export\s+function\s+)([a-zA-Z_][a-zA-Z0-9_]*)"
            class_pattern = r"(?:class|export\s+class)\s+([A-Z][A-Za-z0-9_]*)"
            import_pattern = r"(?:import\s+(?:[\w\{\}\*,\s]+from\s+)?[\"']([a-zA-Z0-9_\-\.\/]+)[\"'])"
            return re.findall(func_pattern, code), re.findall(class_pattern, code), re.findall(import_pattern, code)

        def _extract_html_like(code):
            functions = []
            classes = re.findall(r'class\s*=\s*["\']([^"\']+)["\']', code)
            imports = re.findall(r'<link\s+[^>]*href=["\']([^"\']+)["\']', code)
            return functions, classes, imports

        def _extract_generic(code):
            func_pattern = r"(?:def|function|proc|fn)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
            class_pattern = r"class\s+([A-Z][A-Za-z0-9_]*)"
            import_pattern = r"(?:import|include)\s+([a-zA-Z0-9_\./]+)"
            return re.findall(func_pattern, code), re.findall(class_pattern, code), re.findall(import_pattern, code)

        if language == "Python":
            f, c, i = _extract_python(code)
        elif language in ["JavaScript", "TypeScript", "React (JavaScript)", "React (TypeScript)"]:
            f, c, i = _extract_js_like(code)
        elif language in ["HTML", "CSS"]:
            f, c, i = _extract_html_like(code)
        else:
            f, c, i = _extract_generic(code)

        return sorted(set(f)), sorted(set(c)), sorted(set(i))

    # ================================================================
    # 4️⃣ Async File Analyzer
    # ================================================================
    async def analyze_file(self, path: str):
        language = self.detect_language(path)

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            return {"file_path": path, "error": str(e)}

        functions, classes, imports = self.extract_code_attributes(content, language)
        truncated = content[:4000]

        # Run goal & summary concurrently
        goal_task = self.goal_chain.ainvoke({"filename": os.path.basename(path), "content": truncated})
        summary_task = self.summary_chain.ainvoke({"filename": os.path.basename(path), "content": truncated})
        goal, summary = await asyncio.gather(goal_task, summary_task)

        return {
            "file_path": path,
            "language": language,
            "functions": functions,
            "classes": classes,
            "dependencies": imports,
            "goal": goal,
            "summary": summary,
            "file_content": content[:2000],
        }

    # ================================================================
    # 5️⃣ Recursive Folder Analyzer
    # ================================================================
    async def analyze_folder(self, path: str):
        if os.path.isfile(path):
            return await self.analyze_file(path)

        tasks = []
        for item in os.listdir(path):
            child_path = os.path.join(path, item)
            # skip special / permission-denied folders
            if item in {".git", "node_modules", "__pycache__", ".venv"}:
                continue
            if not os.access(child_path, os.R_OK):
                print(f"⚠️ Skipping {child_path}: permission denied.")
                continue
            tasks.append(self.analyze_folder(child_path))

        if tasks:
            children = await tqdm_asyncio.gather(*tasks, desc=f"📁 {os.path.basename(path)}", leave=False)
        else:
            children = []

        return {"folder_path": path, "children": children}

    # ================================================================
    # 6️⃣ Full Pipeline
    # ================================================================
    async def run(self):
        if not self.repo_dir:
            self.clone_repository()

        print("⚙️ Analyzing repository structure and attributes...")
        result = await self.analyze_folder(self.repo_dir)

        output_file = "repo_attributes.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✅ Attributes extracted and saved to {output_file}")
        return result


# ================================================================
# 7️⃣ Entrypoint (if run standalone)
# ================================================================
if __name__ == "__main__":
    repo_url = "https://github.com/Ebrukoksal/quizzy"
    analyzer = RepositoryAnalyzer(repo_url)
    asyncio.run(analyzer.run())
