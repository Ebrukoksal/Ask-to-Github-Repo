import json
import os
from pyvis.network import Network

def visualize_repo_graph(repo_data_path="repo_attributes.json", output_html="repo_graph.html"):
    """
    Visualizes the output of RepositoryAnalyzer as a knowledge graph.
    """
    if not os.path.exists(repo_data_path):
        raise FileNotFoundError(f"{repo_data_path} not found! Run RepositoryAnalyzer first.")

    with open(repo_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    net = Network(notebook=False, height="750px", width="100%", bgcolor="#0d1117", font_color="white")
    node_counter = 0
    node_map = {}

    def traverse(node, parent_id=None):
        nonlocal node_counter
        if isinstance(node, dict):
            # File node
            if "file_path" in node:
                node_id = node_counter
                node_counter += 1

                label = os.path.basename(node["file_path"])
                tooltip = f"Lang: {node.get('language')}\nFunctions: {len(node.get('functions', []))}\nClasses: {len(node.get('classes', []))}"
                net.add_node(node_id, label=label, title=json.dumps(node, indent=2), shape="box", color="#1f77b4")
                node_map[node["file_path"]] = node_id

                if parent_id is not None:
                    net.add_edge(parent_id, node_id, label="contains", color="#888")

            # Folder node
            elif "folder_path" in node:
                node_id = node_counter
                node_counter += 1
                label = f"📁 {os.path.basename(node['folder_path'])}"
                net.add_node(node_id, label=label, title=node["folder_path"], color="#ff7f0e", shape="ellipse")
                for child in node.get("children", []):
                    traverse(child, parent_id=node_id)

    traverse(data)
    net.save_graph(output_html)
    print(f"✅ Knowledge graph created: {output_html}")
    return output_html
    

