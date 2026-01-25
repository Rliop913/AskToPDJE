from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

PERSIST_DIR = "./rag_db"
STORAGE_DIR = "./storage"
COLLECTION = "my_codebase"

Settings.llm = Ollama(
    model="gpt-oss:20b", temperature=0.1, request_timeout=600.0
)
Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")


def load_index():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_col = client.get_or_create_collection(COLLECTION)
    vec_store = ChromaVectorStore(chroma_collection=chroma_col)
    stor_ctxt = StorageContext.from_defaults(
        persist_dir=STORAGE_DIR, vector_store=vec_store
    )
    return load_index_from_storage(stor_ctxt)


def rag_only_query(query: str, vec_k: int = 8, bm25_k: int = 8, rerank_n: int = 4):
    idx = load_index()

    vec = VectorIndexRetriever(index=idx, similarity_top_k=vec_k)
    bm25 = BM25Retriever.from_defaults(index=idx, similarity_top_k=bm25_k)

    fusion = QueryFusionRetriever(
        retrievers=[vec, bm25],
        similarity_top_k=max(vec_k + bm25_k, 24),
        num_queries=2,
        use_async=False,
    )
    rr = FlagEmbeddingReranker(top_n=rerank_n, model="BAAI/bge-reranker-v2-m3")

    nodes = fusion.retrieve(query)
    reranked = rr.postprocess_nodes(nodes, query=query)
    return reranked


if __name__ == "__main__":
    results = rag_only_query("hello world!")
    for node in results:
        print(node.node.get_content())
