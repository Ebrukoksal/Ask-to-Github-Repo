# main.py
import asyncio
import sys
from helper import RepositoryAnalyzer


class RepoPipeline:
    """
    High-level pipeline:
      • Clone a GitHub repo
      • Analyze code/structure asynchronously
    """

    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.analyzer: RepositoryAnalyzer | None = None
        self.analysis_result = None

    # 1️⃣ CLONE
    def clone(self):
        if self.analyzer is None:
            self.analyzer = RepositoryAnalyzer(self.repo_url)
        print(f"📦 Cloning {self.repo_url} ...")
        repo_path = self.analyzer.clone_repository()
        print(f"✅ Repository cloned at {repo_path}")
        return repo_path

    # 2️⃣ ANALYZE
    async def analyze(self):
        if self.analyzer is None or self.analyzer.repo_dir is None:
            self.clone()
        print("\n🔎 Analyzing repository (this may take a while)...\n")
        # analyzer.run() zaten clone + analyze + JSON yazıyor; sadece analiz istiyorsanız analyze_folder kullanabilirsiniz.
        self.analysis_result = await self.analyzer.analyze_folder(self.analyzer.repo_dir)
        print("\n✅ Analysis complete.")
        return self.analysis_result

    # 3️⃣ RUN FULL PIPELINE
    async def run(self):
        if self.analyzer is None:
            self.analyzer = RepositoryAnalyzer(self.repo_url)
        # RepositoryAnalyzer.run() zaten tüm hattı çalıştırır ve repo_attributes.json yazar
        self.analysis_result = await self.analyzer.run()
        return self.analysis_result


# 4️⃣ CLI ENTRYPOINT
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <REPO_URL>")
        sys.exit(1)

    repo_url = sys.argv[1]
    pipeline = RepoPipeline(repo_url)
    result = asyncio.run(pipeline.run())
    print("\n--- Repository Analysis Result ---\n")
    print(result)
