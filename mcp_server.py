import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from Query import hybrid_query

INDEX_DIRS = (Path("./rag_db"), Path("./storage"))

mcp = FastMCP("AskToPDJE Codebase RAG")


def _index_ready() -> bool:
    return all(path.exists() for path in INDEX_DIRS)


def _node_source(node: Any) -> dict[str, Any]:
    metadata = getattr(node, "metadata", {}) or {}
    file_path = (
        metadata.get("file_path")
        or metadata.get("file_name")
        or metadata.get("filename")
        or metadata.get("path")
    )
    return {
        "path": file_path,
        "score": getattr(node, "score", None),
        "content": node.get_content() if hasattr(node, "get_content") else str(node),
    }


@mcp.tool()
def query_codebase(query: str, top_k: int = 4) -> dict[str, Any]:
    """Search the PDJE/PDJE_Wrapper codebase RAG index and return an answer with sources."""
    if not _index_ready():
        return {
            "error": "Index not found. Run `uv run indexer.py` to build the codebase index."
        }

    qe = hybrid_query(rerank_n=max(top_k, 4))
    response = qe.query(query)
    sources = [
        _node_source(node)
        for node in getattr(response, "source_nodes", [])[:top_k]
    ]
    return {"answer": str(response), "sources": sources}


@mcp.resource("codebase://search/{query}")
def codebase_resource(query: str) -> str:
    """Provide codebase search results as a JSON string resource."""
    return json.dumps(query_codebase(query), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
