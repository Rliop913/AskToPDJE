from pathlib import Path
import os
import shutil
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.node_parser import TokenTextSplitter
from RepoUpdate import clone_or_pull


def update_codebase():
    clone_or_pull("https://github.com/Rliop913/PDJE-Godot-Plugin.git", "./repo/PDJE-Godot-Plugin", "master")
    try:
        os.rmdir("./repo/PDJE-Godot-Plugin/Project-DJ-Engine")
    except:
        pass
    clone_or_pull(
        "https://github.com/Rliop913/Project-DJ-Engine.git", "./repo/PDJE-Godot-Plugin/Project-DJ-Engine"
    )


REPO_ROOT = Path(r"./repo")  # <-- 네 코드베이스 루트로 변경
PERSIST_DIR = "./rag_db"
STORAGE_DIR = "./storage"
COLLECTION = "my_codebase"

class BatchOllamaEmbedding(OllamaEmbedding):
    def _get_text_embeddings(self, texts):
        # texts: List[str]
        # Ollama embed는 input에 리스트를 주면 배치 임베딩 가능
        res = self._client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
        )
        return res.embeddings

    def _get_text_embedding(self, text):
        # 단건도 지원
        res = self._client.embed(
            model=self.model_name,
            input=text,
            options=self.ollama_additional_kwargs,
        )
        return res.embeddings[0]

# 1) LLM / Embedding 세팅
Settings.llm = Ollama(model="gpt-oss:20b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="qwen3-embedding:0.6b")

# 2) 코드/문서 로드 (필요 없는 디렉토리 제외)
EXCLUDE = [
    ".git",
    "SWIG_test",
    "swig_csharp",
    "swig_python",
    "DMCA_FREE_DEMO_MUSIC",
    "build",
    "_deps",
    ".venv",
    "docs",
    "node_modules",
]


def Index(isUpdate: bool):
    update_codebase()
    if not isUpdate:
        ppersist = Path(PERSIST_DIR)
        pstorage = Path(STORAGE_DIR)
        if ppersist.exists():
            shutil.rmtree(ppersist)
            print(f"Removed persist dir: {ppersist}")

        if pstorage.exists():
            shutil.rmtree(pstorage)
            print(f"Removed storage dir: {pstorage}")

        ppersist.mkdir(parents=True, exist_ok=True)
        pstorage.mkdir(parents=True, exist_ok=True)

    docs = SimpleDirectoryReader(
        input_dir=str(REPO_ROOT),
        recursive=True,
        exclude=[f"**/{d}/**" for d in EXCLUDE],
        required_exts=[
            ".cpp",
            ".cc",
            ".c",
            ".hpp",
            ".h",
            ".cmake",
            ".py",
            ".md",
            ".rst",
            ".txt",
            ".yml",
            ".yaml"
            
        ],
        filename_as_id=True,
    ).load_data()

    # 3) 청킹(코드베이스는 chunk_size를 좀 크게 주는 게 보통 유리)
    node_parser = TokenTextSplitter(chunk_size=1200, chunk_overlap=150)

    # 4) Chroma 벡터DB 연결
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = client.get_or_create_collection(COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 5) 인덱스 빌드 + 저장
    if isUpdate:
        storage_ctxt = StorageContext.from_defaults(
            persist_dir=STORAGE_DIR, vector_store=vector_store
        )
        index = load_index_from_storage(storage_ctxt)
        index.refresh_ref_docs(docs)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
    else:
        index = VectorStoreIndex.from_documents(
            docs,
            vector_store=vector_store,
            transformations=[node_parser],
        )
        index.storage_context.persist(persist_dir=STORAGE_DIR)

    print("Indexed docs:", len(docs))
    print("Saved to:", PERSIST_DIR)


if __name__ == "__main__":
    Index(False)
