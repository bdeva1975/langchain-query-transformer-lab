# src/transformers.py
# Implements 4 query transformation techniques from Chapter 9.

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List


def get_llm(config: dict) -> ChatOpenAI:
    return ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"]
    )


# ── Transformer 1: Rewrite-Retrieve-Read ──────────────────────────
def transform_rewrite(query: str, config: dict) -> dict:
    """
    Rewrites a poorly worded query into a clearer,
    more precise form before sending to the retriever.
    """
    llm = get_llm(config)
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant helping to improve search queries.\n\n"
        "Rewrite the following query to be clearer, more specific, "
        "and better suited for document retrieval. "
        "Return ONLY the rewritten query, nothing else.\n\n"
        "Original query: {query}\n\n"
        "Rewritten query:"
    )
    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"query": query})
    return {
        "transformer": "Rewrite-Retrieve-Read",
        "original_query": query,
        "transformed_queries": [rewritten.strip()],
        "retrieval_query": rewritten.strip()
    }


# ── Transformer 2: Multi-Query Generation ─────────────────────────
class MultipleQueries(BaseModel):
    queries: List[str] = Field(
        ..., description="List of query variations"
    )


def transform_multi_query(query: str, config: dict) -> dict:
    """
    Generates multiple variations of the query to retrieve
    a broader and more diverse set of relevant chunks.
    """
    llm = get_llm(config)
    num_queries = config["transformers"]["multi_query"]["num_queries"]

    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant helping to improve document retrieval.\n\n"
        "Generate exactly {num_queries} different variations of the "
        "following query. Each variation should approach the topic "
        "from a slightly different angle to maximize retrieval coverage.\n\n"
        "Original query: {query}\n\n"
        "Return ONLY a numbered list of query variations, nothing else."
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "query": query,
        "num_queries": num_queries
    })

    # Parse numbered list into clean query strings
    lines = [
        line.strip().lstrip("0123456789.-) ").strip()
        for line in result.strip().split("\n")
        if line.strip()
    ]
    queries = [q for q in lines if q][:num_queries]

    return {
        "transformer": "Multi-Query Generation",
        "original_query": query,
        "transformed_queries": queries,
        "retrieval_query": query  # all queries used for retrieval
    }


# ── Transformer 3: Step-Back Questioning ──────────────────────────
def transform_step_back(query: str, config: dict) -> dict:
    """
    Generates a higher-level, more abstract version of the query
    to retrieve broader contextual information.
    """
    llm = get_llm(config)
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant helping to improve document retrieval.\n\n"
        "Given a specific query, generate a more general, higher-level "
        "question that would help retrieve broader context needed to "
        "answer the original query.\n\n"
        "Original query: {query}\n\n"
        "Return ONLY the step-back question, nothing else.\n\n"
        "Step-back question:"
    )
    chain = prompt | llm | StrOutputParser()
    step_back = chain.invoke({"query": query})
    return {
        "transformer": "Step-Back Questioning",
        "original_query": query,
        "transformed_queries": [step_back.strip()],
        "retrieval_query": step_back.strip()
    }


# ── Transformer 4: HyDE ───────────────────────────────────────────
def transform_hyde(query: str, config: dict) -> dict:
    """
    Hypothetical Document Embeddings (HyDE):
    Generates a hypothetical answer to the query and uses
    that as the retrieval query instead of the original question.
    """
    llm = get_llm(config)
    prompt = ChatPromptTemplate.from_template(
        "You are an AI assistant. Write a short hypothetical passage "
        "that would perfectly answer the following question. "
        "Write it as if it were extracted from a real document. "
        "Keep it under 150 words.\n\n"
        "Question: {query}\n\n"
        "Hypothetical passage:"
    )
    chain = prompt | llm | StrOutputParser()
    hypothesis = chain.invoke({"query": query})
    return {
        "transformer": "HyDE",
        "original_query": query,
        "transformed_queries": [hypothesis.strip()],
        "retrieval_query": hypothesis.strip()
    }


# ── No Transformation (baseline) ──────────────────────────────────
def transform_baseline(query: str, config: dict) -> dict:
    """
    Baseline: uses the original query as-is, no transformation.
    """
    return {
        "transformer": "Baseline (No Transformation)",
        "original_query": query,
        "transformed_queries": [query],
        "retrieval_query": query
    }


# ── Registry: all transformers in one place ────────────────────────
TRANSFORMERS = {
    "baseline": transform_baseline,
    "rewrite": transform_rewrite,
    "multi_query": transform_multi_query,
    "step_back": transform_step_back,
    "hyde": transform_hyde,
}