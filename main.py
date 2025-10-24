import asyncio
import sys
from helper import clone_repository, analyze_folder


class RepoPipeline:
    """
    A high-level pipeline that:
      • Clones a GitHub repository
      • Analyzes its code and structure asynchronously
    Acts as a simple orchestrator between helper utilities.
    """

    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.repo_path = None
        self.analysis_result = None

    # ================================================================
    # 1️⃣ CLONE REPOSITORY
    # ================================================================
    def clone(self):
        """Clone the repository and store the path internally."""
        print(f"📦 Cloning {self.repo_url} ...")
        self.repo_path = clone_repository(self.repo_url)
        print(f"✅ Repository cloned at {self.repo_path}")
        return self.repo_path

    # ================================================================
    # 2️⃣ ANALYZE REPOSITORY
    # ================================================================
    async def analyze(self):
        """Analyze the cloned repository asynchronously."""
        if not self.repo_path:
            self.clone()
        print("\n🔎 Analyzing repository (this may take a while)...\n")
        self.analysis_result = await analyze_folder(self.repo_path)
        print("\n✅ Analysis complete.")
        return self.analysis_result

    # ================================================================
    # 3️⃣ RUN FULL PIPELINE
    # ================================================================
    async def run(self):
        """Full pipeline: clone + analyze."""
        self.clone()
        await self.analyze()
        return self.analysis_result


# ================================================================
# 4️⃣ CLI ENTRYPOINT
# ================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <REPO_URL>")
        sys.exit(1)

    repo_url = sys.argv[1]
    pipeline = RepoPipeline(repo_url)
    result = asyncio.run(pipeline.run())
    print("\n--- Repository Analysis Result ---\n")
    print(result)
