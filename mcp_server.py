import json
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from rag_only_query import rag_only_query

INDEX_DIRS = (Path("./rag_db"), Path("./storage"))

mcp = FastMCP("AskToPDJE Codebase RAG")


def _index_ready() -> bool:
    return all(path.exists() for path in INDEX_DIRS)


def _node_source(node) -> dict[str, Any]:
    # node: NodeWithScore 같은 타입이라고 가정
    n = node.node if hasattr(node, "node") else node
    md = getattr(n, "metadata", {}) or {}

    path = md.get("file_path") or md.get("path") or "unknown"
    start = md.get("start_line") or md.get("line_start") or md.get("start") or "?"
    end = md.get("end_line") or md.get("line_end") or md.get("end") or "?"

    text = ""
    if hasattr(n, "get_content"):
        text = n.get_content() or ""
    text = text.strip()
    text = text[:600]  # ✅ 너무 길면 Continue에서 잘림/깨짐 유발

    score = getattr(node, "score", None)
    return {"path": path, "start": start, "end": end, "score": score, "snippet": text}

def _format_sources_md(query: str, sources: list[dict[str, Any]]) -> str:
    lines = [f"### Codebase search: `{query}`", ""]
    if not sources:
        return "\n".join(lines + ["(no results)"])

    for i, s in enumerate(sources, 1):
        lines.append(f"**{i}) {s['path']}:{s['start']}-{s['end']}**  (score={s['score']})")
        lines.append("```")
        lines.append(s["snippet"])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)

@mcp.tool()
def query_codebase(query: str, top_k: int = 4) -> dict[str, Any]:
    """Search the PDJE/PDJE_Wrapper codebase RAG index and return an answer with sources."""
    if not _index_ready():
        return {
            "error": "Index not found. Run `uv run indexer.py` to build the codebase index."
        }

    nodes = rag_only_query(query, rerank_n=max(top_k, 4))
    sources = [_node_source(node) for node in nodes[:top_k]]
    return _format_sources_md(query, sources)


@mcp.resource("codebase://search/{query}")
def codebase_resource(query: str) -> str:
    """Provide codebase search results as a JSON string resource."""
    return json.dumps(query_codebase(query), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
