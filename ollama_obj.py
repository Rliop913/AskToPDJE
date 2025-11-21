from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFDirectoryLoader


from langchain_community.vectorstores import FAISS  # Vector DB
from langchain_community.embeddings import OllamaEmbeddings  # set embed model
from langchain.schema.runnable import RunnableMap
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
import json


class ollamas:
    def __init__(
        self,
        embed_model="nomic-embed-text",
        #  stt_model="large-v3",
        llm_model="gemma3n:e4b",
    ):
        self.embedder = OllamaEmbeddings(model=embed_model)

        self.LLM = ChatOllama(model=llm_model)

        self.chain = None
        self.VDB = None
        self.FLAG_VDB_IS_EMPTY = True
        self.char_prompt: str = ""

    def text_chat(self, msg: str):
        response = self.chain.invoke({"charText": self.char_prompt, "talk": msg})
        tempdata = dict()
        tempdata["TYPE"] = "chat_result"
        tempdata["DATA"] = response
        return tempdata

    # def load_VDB(self, memory_length=10, read_documents=5):
    #     try:
    #         self.VDB = FAISS.load_local("VDB_FOR_RAG", self.embedder, allow_dangerous_deserialization=True)
    #         self.FLAG_VDB_IS_EMPTY=False
    #     except:
    #         pass
    #     self.init_chain(memory_length, read_documents)

    # def remake_VDB(self, document_path:str):
    #     pdfs_loaded = PyPDFDirectoryLoader(document_path, mode="page")
    #     pdf_docs = pdfs_loaded.load()
    #     chunker = RecursiveCharacterTextSplitter(
    #         separators=["\n\n", "\n", ".", " "],
    #         chunk_size=300,
    #         chunk_overlap=50
    #     )
    #     chunked = chunker.split_documents(pdf_docs)
    #     self.VDB = FAISS.from_documents(chunked, self.embedder)
    #     self.VDB.save_local("VDB_FOR_RAG")
    #     self.FLAG_VDB_IS_EMPTY = False

    def init_chain(self, memory_length=10, read_documents=5):
        if self.FLAG_VDB_IS_EMPTY:
            self.promt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("history"),
                    ("system", "This is your role: {charText}"),
                    ("user", "{talk}"),
                ]
            )
            self.memory = ConversationBufferWindowMemory(
                k=memory_length, return_messages=True
            )

            self.chain = (
                RunnableMap(
                    {
                        "history": lambda x: self.memory.chat_memory.messages,
                        "charText": lambda x: x["charText"],
                        "talk": lambda x: x["talk"],
                    }
                )
                | self.promt
                | self.LLM
                | StrOutputParser()
            )
        else:
            self.promt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder("history"),
                    (
                        "system",
                        "This is from documents about you. \n\n {context} \n\n This is your role: {charText}",
                    ),
                    ("user", "{talk}"),
                ]
            )
            self.memory = ConversationBufferWindowMemory(
                k=memory_length, return_messages=True
            )
            self.VDBResponser = self.VDB.as_retriever(
                search_kwargs={"k": read_documents}
            )  # get top 2 documents
            self.chain = (
                RunnableMap(
                    {
                        "context": lambda x: self.VDBResponser,
                        "history": lambda x: self.memory.chat_memory.messages,
                        "charText": lambda x: x["charText"],
                        "talk": lambda x: x["talk"],
                    }
                )
                | self.promt
                | self.LLM
                | StrOutputParser()
            )
