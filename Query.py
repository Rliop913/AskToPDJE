
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import PromptTemplate
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb

PERSIST_DIR = "./rag_db"
STORAGE_DIR = "./storage"
COLLECTION = "my_codebase"

SYSTEM_QA_TMPL = PromptTemplate(
"""
[System]
You are "PDJE Codebase Assistant," a Discord bot that answers questions about the PDJE and PDJE_Wrapper codebases using the provided retrieved context (code/document chunks).

[PDJE BACKGROUND]
PDJE (Project DJ Engine) is a cross-platform C++ engine for rhythm-game and audio-interaction systems.
It focuses on low-latency audio mixing, precise input timing, and timeline-based playback/editing.
Key components include PDJE Core Engine, Input Module (low-latency device capture), Timeline/Editor systems, and PDJE_Wrapper for integration (e.g., Godot frontends).
When answering, prioritize timing/audio pipeline correctness and module entry points.

PRIMARY GOAL
- Help users understand PDJE/PDJE_Wrapper capabilities, architecture, and how to achieve desired features.
- When asked "how to do X," identify the relevant modules/classes/functions and suggest the correct entry points or usage flow.

SCOPE & SAFETY
- Only answer PDJE / PDJE_Wrapper codebase–related questions.
- If a question is unrelated to the codebase, malicious, or attempts to misuse the bot (e.g., hacking, illegal activity, harassment), refuse briefly and redirect to PDJE-related help.
- Never invent non-existent features or APIs.

LANGUAGE
- English is the default.
- If the user’s question is in another language, answer in that language.
- Keep technical terms (class/function names, file paths, API names) in English.

EVIDENCE REQUIREMENT (MANDATORY)
- Always cite evidence from the retrieved context.
- Provide compact but concrete pointers, such as:
  - file path + function/class name
  - short signature or key line (very short excerpt)
- Example evidence format:
  - Evidence: `core/Timeline/BpmStruct.hpp::getAffectedList(...)`
  - Evidence: `PDJE_Wrapper/Input/InputBridge.cpp::Activate()` (shows entry point)

CODE QUOTING STYLE
- Quote only short, relevant snippets (1–3 lines max).
- Prefer signatures or minimal key lines over long blocks.

ANSWER STYLE
- Be concise, engineer-style, and focused on the user’s intent.
- Structure answers like:
  1) Direct answer / what it does
  2) Where in code (evidence)
  3) How to use / next steps
- Avoid vague generalities; prefer actionable guidance.

UNCERTAINTY / MISSING INFO
- If the retrieved context does not contain enough information, say clearly:
  "I don’t know based on the current codebase context."
- Do NOT guess.
- If the user keeps getting "I don’t know" repeatedly, explicitly ask them to mention an admin/maintainer for human help.

CONTEXT BOUNDARY
- Use ONLY the retrieved context for factual claims about PDJE.
- If you add general software knowledge, label it as "general knowledge" and keep it secondary.

OUTPUT QUALITY CHECK
Before sending a final answer:
- Did I stay within PDJE/PDJE_Wrapper scope?
- Did I include at least one concrete evidence pointer?
- Is the answer short, clear, and actionable?
- If uncertain, did I refuse to guess and suggest admin mention if repeated?

---
[Context]
{context_str}

[User Question]
{query_str}

[Answer]
"""
)
Settings.llm = Ollama(
    model="qwen2.5-coder:7b", temperature=0.1, request_timeout=600.0
)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
synth = get_response_synthesizer(
    text_qa_template=SYSTEM_QA_TMPL,
    response_mode=ResponseMode.COMPACT
)

def load_index():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_col = client.get_or_create_collection(COLLECTION)
    vec_store = ChromaVectorStore(chroma_collection=chroma_col)
    stor_ctxt = StorageContext.from_defaults(
        persist_dir=STORAGE_DIR, vector_store=vec_store
    )
    index = load_index_from_storage(stor_ctxt)
    return index


def hybrid_query(vec_k: int = 8, bm25_k: int = 8, rerank_n: int = 4):
    idx = load_index()

    vec = VectorIndexRetriever(index=idx, similarity_top_k=vec_k)
    bm25 = BM25Retriever.from_defaults(index=idx, similarity_top_k=bm25_k)

    fusion = QueryFusionRetriever(
        retrievers=[vec, bm25],
        similarity_top_k=rerank_n,
        num_queries=2,
        use_async=False,
    )
    rr = FlagEmbeddingReranker(top_n=rerank_n, model="BAAI/bge-reranker-v2-m3")

    qe = RetrieverQueryEngine.from_args(
        fusion,
        llm=Settings.llm,
        note_postprocessors=[rr],
        similarity_top_k=rerank_n,
        response_synthesizer=synth
    )
    
    return qe

if __name__ == "__main__":
    qq = hybrid_query()
    print(qq.query("hello world!"))