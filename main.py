# main.py
import asyncio
import sys
from helper import RepositoryAnalyzer
from knowledge_graph import visualize_repo_graph

class RepoPipeline:
    def __init__(self, repo_url: str):
        self.repo_url = repo_url
        self.analyzer = RepositoryAnalyzer(repo_url)
        self.analysis_result = None

    def clone(self):
        print(f"📦 Cloning {self.repo_url} ...")
        repo_path = self.analyzer.clone_repository()
        print(f"✅ Repository cloned at {repo_path}")
        return repo_path

    

    async def analyze(self):
        if not self.analyzer.repo_dir:
            self.clone()
        print("\n🔎 Analyzing repository...\n")
        self.analysis_result = await self.analyzer.analyze_folder(self.analyzer.repo_dir)
        print("\n✅ Analysis complete.")
        return self.analysis_result

    async def run(self):
        self.clone()
        await self.analyze()
        visualize_repo_graph(repo_data_path="repo_attributes.json")
        return self.analysis_result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <REPO_URL>")
        sys.exit(1)
    repo_url = sys.argv[1]
    pipeline = RepoPipeline(repo_url)
    result = asyncio.run(pipeline.run())
    print("\n--- Repository Analysis Result ---\n")
    print(result)
