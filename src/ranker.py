from transformers import AutoTokenizer, AutoModel
from typing import Optional, List, Tuple
import torch


class ReRanker:
    def __init__(
        self,
        model: str = "colbert-ir/colbertv2.0",
        tokenizer: str = "colbert-ir/colbertv2.0",
        device: Optional[str] = None,
        DEFAULT_COLBERT_MAX_LENGTH=512,
    ):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._model = AutoModel.from_pretrained(model)
        self._device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.mps.is_available() else "cpu"
        )
        self._model.to(self._device)
        self.DEFAULT_COLBERT_MAX_LENGTH = DEFAULT_COLBERT_MAX_LENGTH

    def _calculate_sim(self, query: str, documents_text_list: List[str]) -> List[float]:
        query_encoding = self._tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.DEFAULT_COLBERT_MAX_LENGTH  # Which is set to 512!
        )
        
        query_encoding = {k: v.to(self._device) for k, v in query_encoding.items()}
        
        query_embedding = self._model(
            **query_encoding
        ).last_hidden_state  # [1, query_len, embed_dim]

        rerank_score_list = []

        for document_text in documents_text_list:
            document_encoding = self._tokenizer(
                document_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.DEFAULT_COLBERT_MAX_LENGTH
            )
            document_encoding = {k: v.to(self._device) for k, v in document_encoding.items()}

            document_embedding = self._model(
                **document_encoding
            ).last_hidden_state  # [1, doc_len, embed_dim]

            # Compute similarity matrix between query tokens and doc tokens
            sim_matrix = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(2),  # [1, query_len, 1, embed_dim]
                document_embedding.unsqueeze(1),  # [1, 1, doc_len, embed_dim]
                dim=-1,
            )  # [1, query_len, doc_len]

            max_sim_scores, _ = torch.max(sim_matrix, dim=2)  # [1, query_len]
            avg_score = torch.mean(max_sim_scores, dim=1)  # [1]
            rerank_score_list.append(avg_score.item())  # Append as float

        return rerank_score_list

    def rerank(
        self, query: str, documents: List, top_k: int = 3
    ) -> List[Tuple[object, float]]:
        """
        Rerank a list of Document objects based on their page_content,
        returning the top_k Documents along with their scores.

        Args:
            query (str): The input query.
            documents (List): A list of Document objects, each having page_content and metadata.
            top_k (int): Number of top results to return.

        Returns:
            List[Tuple[Document, float]]: Top-k documents with their relevance scores.
        """

        # Extract page content for scoring
        document_contents = [doc.page_content for doc in documents]

        # Calculate similarity scores based on content
        scores = self._calculate_sim(query, document_contents)

        # Pair Document objects with their scores
        scored_docs = list(zip(documents, scores))

        # Sort by score in descending order
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)

        return sorted_docs[:top_k]
