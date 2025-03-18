# bring in deps
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
import tiktoken
import sqlalchemy
import enum
import pickle
import getpass
import os

from src.ranker import ReRanker

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


class HNSWDistanceStrategy(str, enum.Enum):
    """Enumerator of the HNSW index Distance strategies."""

    EUCLIDEAN = "vector_l2_ops"
    COSINE = "vector_cosine_ops"
    MAX_INNER_PRODUCT = "vector_ip_ops"
    L1_DISTANCE = "vector_l1_ops"
    HAMMING_DISTANCE = "bit_hamming_ops"
    Jaccard_DISTANCE = "bit_jaccard_ops"


class HNSWIndexing:
    """
    Implementation fo HNSW indexing method. Which is a workaround method
    of https://github.com/langchain-ai/langchain-postgres/pull/85
    """

    ADA_TOKEN_COUNT = 1536
    DEFAULT_HNSW_DISTANCE_STRATEGY = HNSWDistanceStrategy.COSINE

    def __init__(self, session):
        self.session = session
        self._execute_hnsw_settings()

    def _execute_hnsw_settings(self) -> None:
        self.session.execute(sqlalchemy.text("SET LOCAL enable_seqscan = off;"))
        self.session.execute(sqlalchemy.text(f"SET LOCAL hnsw.ef_search = 100;"))
        self.session.commit()

    def _prepare_create_hnsw_index_query(
        self,
        dims: int = ADA_TOKEN_COUNT,
        distance_strategy: HNSWDistanceStrategy = DEFAULT_HNSW_DISTANCE_STRATEGY,
        m: int = 8,
        ef_construction: int = 16,
    ) -> sqlalchemy.TextClause:
        create_index_query = sqlalchemy.text(
            "CREATE INDEX IF NOT EXISTS langchain_pg_embedding_idx "
            "ON langchain_pg_embedding USING hnsw ((embedding::vector({})) {}) "
            "WITH ("
            "m = {}, "
            "ef_construction = {}"
            ");".format(dims, distance_strategy.value, m, ef_construction)
        )

        return create_index_query

    def create_hnsw_index(
        self,
        distance_strategy: HNSWDistanceStrategy = DEFAULT_HNSW_DISTANCE_STRATEGY,
        m: int = 8,
        ef_construction: int = 16,
    ) -> None:
        assert self.session, "engine not found"
        create_index_query = self._prepare_create_hnsw_index_query(
            distance_strategy=distance_strategy, m=m, ef_construction=ef_construction
        )

        # Execute the queries
        try:
            self.session.execute(create_index_query)
            self.session.commit()
            print("HNSW extension and index created successfully.")  # noqa: T201
        except Exception as e:
            print(f"Failed to create HNSW extension or index: {e}")  # noqa: T201
            raise e


class VectorDB:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
        self.collection_name = "my_docs"
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.collection_name,
            connection=self.connection,
            use_jsonb=True,
        )
        self.reranker = ReRanker()

    def enable_hnsw_indexing(self):
        hnsw = HNSWIndexing(self.vector_store.session_maker)
        hnsw.create_hnsw_index()

    def add_documents(self, documents, ignore_document_by_similarity_threshold=0.8):
        docs_to_add = []
        for doc in documents:
            # Search for similar documents in the store
            # similar_docs = self.similarity_search(
            #     doc.page_content, top_k=1, initial_k=1
            # )

            # score = similar_docs[0][1]

            # if score >= ignore_document_by_similarity_threshold:
            #     print(
            #         f"Document with id {doc.metadata['id']} already present. Skipping."
            #     )
            #     continue

            docs_to_add.append(doc)

        if docs_to_add:
            self.vector_store.add_documents(
                docs_to_add, ids=[doc.metadata["id"] for doc in docs_to_add]
            )

    def similarity_search(self, query, top_k=3, initial_k=10):
        docs = self.vector_store.similarity_search(query, k=initial_k)
        reranked_docs = self.reranker.rerank(query, docs, top_k=top_k)
        return reranked_docs

    def truncate_text_by_tokens(self, text, max_tokens=512, model_name="text-embedding-3-small"):
        tokenizer = tiktoken.encoding_for_model(model_name)
        tokens = tokenizer.encode(text)
        truncated_tokens = tokens[:max_tokens]
        return tokenizer.decode(truncated_tokens)


class ExtractDocs:
    def __init__(self, document_type="pdf"):
        self.documents = None
        self.document_type = document_type

    def extract(self, file_path):
        # set up parser
        parser = LlamaParse(result_type="text")  # "markdown" and "text" are available

        # use SimpleDirectoryReader to parse our file
        file_extractor = {f".{self.document_type}": parser}
        self.documents = SimpleDirectoryReader(
            input_files=[file_path], file_extractor=file_extractor
        ).load_data()

        embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )

        nodes = splitter.get_nodes_from_documents(self.documents)

        docs = list()
        for doc in nodes:
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


if __name__ == "__main__":
    # Load the documents
    docs = ExtractDocs().extract("data/Constitution_of_the_Republic_of_Singapore.pdf")
    # Save the documents
    db = VectorDB()
    db.add_documents(docs)
    db.enable_hnsw_indexing()
