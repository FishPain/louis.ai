# louis.ai

## Description

An implementation of an agentic paralegal using advanced RAG retrieval and ranking techniques.

## Installation

```bash
pip install -r requirements.txt

sudo chmod 777 create_vectordb.sh
./create_vectordb.sh

echo "run the scripts within main.ipynb"
```

## Technologies

- OpenAI GPT-3.5-turbo-0125
- OpenAI text-embedding-3-large
- Langchain
- Langgraph
- Langsmith
- pgvector
- LlamaCloud
- TAVILY

## Methodology

### RAG

- Ranking: [RankNet, LambdaRank, LambdaMART](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
- [ColBERT](https://arxiv.org/pdf/2004.12832)
- Retrieval: [RAPTOR](https://arxiv.org/html/2401.18059v1), [Multi-Representation-Indexing](https://www.linkedin.com/posts/langchain_rag-from-scratch-multi-representation-activity-7179205407217766400-Wm0w/),
