from pathlib import Path
import argparse
import os
import shutil
import time
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
    StorageContext,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core.node_parser import TokenTextSplitter
from RepoUpdate import clone_or_pull
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


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
REQUIRED_EXTS = [
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
    ".yaml",
]
NODE_PARSER = TokenTextSplitter(chunk_size=1200, chunk_overlap=150)
Settings.node_parser = NODE_PARSER


def _ensure_storage_dirs():
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


def _build_vector_store():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    chroma_collection = client.get_or_create_collection(COLLECTION)
    return ChromaVectorStore(chroma_collection=chroma_collection)


def _load_docs(repo_root: Path):
    return SimpleDirectoryReader(
        input_dir=str(repo_root),
        recursive=True,
        exclude=[f"**/{d}/**" for d in EXCLUDE],
        required_exts=REQUIRED_EXTS,
        filename_as_id=True,
    ).load_data()


def _load_doc_from_file(file_path: Path):
    return SimpleDirectoryReader(
        input_files=[str(file_path.resolve())],
        exclude=[f"**/{d}/**" for d in EXCLUDE],
        required_exts=REQUIRED_EXTS,
        filename_as_id=True,
    ).load_data()


def _should_index(path: Path, repo_root: Path) -> bool:
    if not path.exists() or path.is_dir():
        return False
    if path.suffix.lower() not in REQUIRED_EXTS:
        return False
    try:
        rel_parts = path.relative_to(repo_root).parts
    except ValueError:
        return False
    return not any(part in EXCLUDE for part in rel_parts)


def Index(isUpdate: bool, repo_root: Path, skip_repo_update: bool = False):
    if not skip_repo_update:
        update_codebase()
    if not isUpdate:
        _ensure_storage_dirs()

    docs = _load_docs(repo_root)
    vector_store = _build_vector_store()

    if isUpdate and Path(STORAGE_DIR).exists():
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
            transformations=[NODE_PARSER],
        )
        index.storage_context.persist(persist_dir=STORAGE_DIR)

    print("Indexed docs:", len(docs))
    print("Saved to:", PERSIST_DIR)
    return index


def _load_or_create_index(repo_root: Path, skip_repo_update: bool):
    if Path(STORAGE_DIR).exists() and Path(PERSIST_DIR).exists():
        vector_store = _build_vector_store()
        storage_ctxt = StorageContext.from_defaults(
            persist_dir=STORAGE_DIR, vector_store=vector_store
        )
        return load_index_from_storage(storage_ctxt)
    return Index(False, repo_root=repo_root, skip_repo_update=skip_repo_update)


class IndexWatchHandler(FileSystemEventHandler):
    def __init__(self, index, repo_root: Path):
        self.index = index
        self.repo_root = repo_root

    def _refresh_file(self, file_path: Path):
        if not _should_index(file_path, self.repo_root):
            return
        docs = _load_doc_from_file(file_path.resolve())
        if not docs:
            return
        self.index.refresh_ref_docs(docs)
        self.index.storage_context.persist(persist_dir=STORAGE_DIR)
        print(f"Updated index for: {file_path}")

    def on_created(self, event):
        if not event.is_directory:
            self._refresh_file(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory:
            self._refresh_file(Path(event.src_path))

    def on_deleted(self, event):
        if event.is_directory:
            return
        file_path = Path(event.src_path)
        try:
            self.index.delete_ref_doc(
                str(file_path.resolve()),
                delete_from_docstore=True,
            )
            self.index.storage_context.persist(persist_dir=STORAGE_DIR)
            print(f"Removed from index: {file_path}")
        except Exception as exc:
            print(f"Failed to delete {file_path}: {exc}")

    def on_moved(self, event):
        if event.is_directory:
            return
        src_path = Path(event.src_path)
        dest_path = Path(event.dest_path)
        try:
            self.index.delete_ref_doc(
                str(src_path.resolve()),
                delete_from_docstore=True,
            )
        except Exception as exc:
            print(f"Failed to delete {src_path}: {exc}")
        self._refresh_file(dest_path)


def watch_index(repo_root: Path, skip_repo_update: bool):
    if not skip_repo_update:
        update_codebase()
    index = _load_or_create_index(repo_root, skip_repo_update=skip_repo_update)
    handler = IndexWatchHandler(index, repo_root=repo_root)
    observer = Observer()
    observer.schedule(handler, str(repo_root), recursive=True)
    observer.start()
    print(f"Watching for changes under: {repo_root}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index PDJE codebase for RAG.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="Incrementally refresh the existing index.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for file changes and update the index in real time.",
    )
    parser.add_argument(
        "--skip-repo-update",
        action="store_true",
        help="Skip git clone/pull and index the local repo directly.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Path to the codebase to index.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args.watch:
        watch_index(repo_root, skip_repo_update=args.skip_repo_update)
    else:
        Index(
            args.update,
            repo_root=repo_root,
            skip_repo_update=args.skip_repo_update,
        )
