from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import chromadb
from langchain_community.retrievers.llama_index import LlamaIndexRetriever
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama
PERSIST_DIR = "./rag_db"
STORAGE_DIR = "./storage"
COLLECTION = "my_codebase"

Settings.llm = Ollama(model="qwen2.5-coder:7b", temperature=0.1)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

client = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = client.get_or_create_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(
    persist_dir=STORAGE_DIR,
    vector_store=vector_store,
)
index = load_index_from_storage(storage_context)
# 벡터 retriever
vec_retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

nodes = list(index.docstore.docs.values())
# BM25 retriever (nodes 기반)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10
)

# RRF/Relative Score Fusion
fusion_retriever = QueryFusionRetriever(
    retrievers=[bm25_retriever, vec_retriever],
    similarity_top_k=4,
    num_queries=1,
    mode="reciprocal_rerank",  # RRF
)

reranker__ = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-v2-m3",  # 로컬 HF 모델 경로도 가능
    top_n=3
)

# query_engine = index.as_query_engine(
#     fusion_retriever,                 # ← retriever를 위치 인자로
#     node_postprocessors=[reranker],
# )

from typing import List
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
# LlamaIndex에서 만든 fusion_retriever / reranker를 그대로 주입
class LlamaIndexHybridRetriever(BaseRetriever):
    def __init__(self, __fusion_retriever, __reranker):
        super().__init__()
        self.__fusion_retriever = __fusion_retriever
        self.__reranker = __reranker

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 1) LlamaIndex 하이브리드 retrieval
        nodes = self.__fusion_retriever.retrieve(query)

        # 2) LlamaIndex reranker 적용
        if self.__reranker is not None:
            nodes = self.__reranker.postprocess_nodes(nodes=nodes, query_str=query)

        # 3) LangChain Document로 변환
        docs = []
        for n in nodes:
            meta = dict(n.metadata or {})
            docs.append(
                Document(
                    page_content=n.get_content(),
                    metadata=meta
                )
            )
        return docs
    
    
lc_retriever = LlamaIndexHybridRetriever(fusion_retriever, reranker__)

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

llm = ChatOllama(model="qwen2.5-coder:7b", temperature=0.1)

# qa = RetrievalQA.from_chain_type(
#     llm=Settings.llm,
#     retriever=lc_retriever,
#     return_source_documents=True,
# )

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about the codebase using ONLY the context below.\n\n"
     "Context:\n{context}"),
    ("human", "{input}")
])

doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(lc_retriever, doc_chain)

# 4) 질의
res = rag_chain.invoke({"input": "PDJE 인터페이스에 어떤 함수가 있는지 알려줘."})
print(res["answer"])