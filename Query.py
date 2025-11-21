
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
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


def load_index():
    Settings.llm = Ollama(
        model="qwen2.5-coder:7b", temperature=0.1, request_timeout=600.0
    )
    Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_col = client.get_or_create_collection(COLLECTION)
    vec_store = ChromaVectorStore(chroma_collection=chroma_col)
    stor_ctxt = StorageContext.from_defaults(
        persist_dir=STORAGE_DIR, vector_store=vec_store
    )
    index = load_index_from_storage(stor_ctxt)
    return index


def hybrid_query(vec_k: int = 8, bm25_k: int = 8, rerank_n: int = 6):
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
    )
    
    return qe

if __name__ == "__main__":
    qq = hybrid_query()
    print(qq.query("hello world!"))