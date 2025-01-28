# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
import pickle
import getpass
import os


if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


class VectorDB:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
        self.collection_name = "my_docs"
        self.vector_store = None

    def create_vector_store(self):
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
        )
        return self.vector_store

    def add_documents(self, documents):
        self.vector_store.add_documents(
            documents, ids=[doc.metadata["id"] for doc in self.documents]
        )

    def similarity_search(self, query, k=3):
        docs = self.vector_store.similarity_search(query, k=k)
        return docs


class ExtractDocs:
    def __init__(self, documents):
        self.documents = documents

    @staticmethod
    def extract(file_path):
        # set up parser
        parser = LlamaParse(
            result_type="markdown"  # "markdown" and "text" are available
        )

        # use SimpleDirectoryReader to parse our file
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            input_files=[file_path], file_extractor=file_extractor
        ).load_data()

        docs = list()
        for doc in self.documents:
            docs.append(Document(page_content=doc.text, metadata={"id": doc.id_}))
        return docs

    def save(self):
        # Assuming 'docs' is the variable containing the documents you want to save
        with open("data/constitution.pkl", "wb") as file:
            pickle.dump(self.documents, file)

    def load(self):
        # Assuming 'docs' is the variable containing the documents you want to save
        with open("data/constitution.pkl", "rb") as file:
            docs = pickle.load(file)
        return docs
