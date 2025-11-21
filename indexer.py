from pathlib import Path
import os

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

REPO_ROOT = Path(r"./repos/Project-DJ-Engine")   # <-- 네 코드베이스 루트로 변경
PERSIST_DIR = "./rag_db"
STORAGE_DIR = "./storage"
COLLECTION = "my_codebase"

# 1) LLM / Embedding 세팅
Settings.llm = Ollama(model="gemma3n:e4b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="mxbai-embed-large")

# 2) 코드/문서 로드 (필요 없는 디렉토리 제외)
EXCLUDE = [".git","docs","document_sources","SWIG_test","swig_csharp","swig_python","DMCA_FREE_DEMO_MUSIC", "build", "_deps", ".venv", "extern", "node_modules"]
docs = SimpleDirectoryReader(
    input_dir=str(REPO_ROOT),
    recursive=True,
    exclude=[f"**/{d}/**" for d in EXCLUDE],
    required_exts=[
        ".cpp",".cc",".c",".hpp",".h",".cmake",
        ".py",".md",".rst",".txt",".yml",".yaml",".json"
    ],
).load_data()

# 3) 청킹(코드베이스는 chunk_size를 좀 크게 주는 게 보통 유리)
node_parser = SentenceSplitter(chunk_size=1200, chunk_overlap=150)

# 4) Chroma 벡터DB 연결
client = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = client.get_or_create_collection(COLLECTION)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 5) 인덱스 빌드 + 저장
index = VectorStoreIndex.from_documents(
    docs,
    vector_store=vector_store,
    transformations=[node_parser],
)
index.storage_context.persist(persist_dir=STORAGE_DIR)
print("Indexed docs:", len(docs))
print("Saved to:", PERSIST_DIR)