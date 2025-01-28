from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator

from langchain_core.prompts import ChatPromptTemplate

from src.constant import Routing

def decide_action(query, chat_model):
    """
    Uses an LLM to decide whether to use tools or rely on RAG.
    Returns either 'vectorstore' or 'web_search'.
    """

    class ChooseScope(BaseModel):
        datasource: Literal[Routing.VECTORSTORE, Routing.WEB_SEARCH] = Field(
            description="Given a user question choose to route it to web search or a vectorstore."
        )

    decision_prompt = "You are an intelligent assistant specializing in legal queries. Your job is to decide whether the user's query \
        requires an external search using tools (e.g., searching for the latest case laws or regulations) or can be \
        answered using the retrieved legal documents provided."

    structured_output_parser = chat_model.with_structured_output(ChooseScope)

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", decision_prompt),
            ("human", "{question}"),
        ]
    )

    prompt_and_model = route_prompt | structured_output_parser
    decision_response = prompt_and_model.invoke({"question": query})
    return decision_response.datasource


def get_chatgpt_response(query, retrieved_docs):
    # Initialize model and tools
    chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    search = TavilySearchResults(max_results=2)
    tools = [search]

    # Step 1: Decide action using LLM
    action = decide_action(query, chat_model)

    if action == Routing.WEB_SEARCH:
        agent_executor = create_react_agent(chat_model, tools)
        # Use tools to perform external search
        response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
        return response

    elif action == Routing.VECTORSTORE:
        # Use RAG (retrieved documents)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        rag_prompt = f"""
        You are a highly skilled legal expert specializing in Singaporean law, with extensive experience in drafting contracts, \
        interpreting legislation, and providing sound legal advice. Using the following retrieved legal context, provide a clear, \
        concise, and accurate response to the user's query. Where necessary, reference specific clauses or legal principles mentioned in the context.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:
        """
        response = chat_model.invoke([HumanMessage(content=rag_prompt)])
        print(f"Using RAG Response: {response.content}")
        return response
    else:
        raise ValueError(f"Invalid decision: {action}")
