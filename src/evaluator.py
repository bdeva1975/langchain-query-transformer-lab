# src/evaluator.py
# Generates answers and scores each transformer's retrieval quality.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np


def get_answer(query: str, docs: list, config: dict) -> str:
    """Generate an answer using retrieved docs as context."""
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"]
    )
    prompt = ChatPromptTemplate.from_template(
        "Answer the following question based only on the "
        "provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "query": query})


def score_relevance(
    query: str,
    docs: list,
    config: dict
) -> float:
    """
    Cosine similarity between query embedding and
    combined context embedding.
    """
    if not docs:
        return 0.0

    embeddings = OpenAIEmbeddings(model=config["embeddings"]["model"])
    context = "\n\n".join([doc.page_content for doc in docs])

    q_vec = np.array(embeddings.embed_query(query))
    c_vec = np.array(embeddings.embed_query(context[:2000]))

    cosine = np.dot(q_vec, c_vec) / (
        np.linalg.norm(q_vec) * np.linalg.norm(c_vec)
    )
    return round(float(cosine), 4)


def score_faithfulness(
    answer: str,
    docs: list,
    config: dict
) -> float:
    """
    LLM-judged score: is the answer grounded in the context?
    """
    if not docs:
        return 0.0

    context = "\n\n".join([doc.page_content for doc in docs])
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=0
    )
    prompt = ChatPromptTemplate.from_template(
        "You are an evaluation assistant.\n\n"
        "Score how faithful the answer is to the context "
        "on a scale from 0.0 to 1.0.\n"
        "0.0 = answer is not supported by context at all.\n"
        "1.0 = answer is fully supported by context.\n\n"
        "Context:\n{context}\n\n"
        "Answer:\n{answer}\n\n"
        "Return ONLY a single decimal number. No explanation."
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({
        "context": context[:2000],
        "answer": answer
    })
    try:
        return round(float(result.strip()), 4)
    except ValueError:
        return 0.0


def score_completeness(
    query: str,
    answer: str,
    config: dict
) -> float:
    """
    LLM-judged score: does the answer fully address the question?
    """
    llm = ChatOpenAI(
        model=config["llm"]["model"],
        temperature=0
    )
    prompt = ChatPromptTemplate.from_template(
        "You are an evaluation assistant.\n\n"
        "Score how completely the answer addresses the question "
        "on a scale from 0.0 to 1.0.\n"
        "0.0 = answer does not address the question at all.\n"
        "1.0 = answer fully addresses the question.\n\n"
        "Question:\n{query}\n\n"
        "Answer:\n{answer}\n\n"
        "Return ONLY a single decimal number. No explanation."
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"query": query, "answer": answer})
    try:
        return round(float(result.strip()), 4)
    except ValueError:
        return 0.0


def evaluate(
    transformer_name: str,
    original_query: str,
    transformed_queries: list,
    docs: list,
    config: dict
) -> dict:
    """
    Run full evaluation for one transformer on one query.
    Returns a result row ready for display and CSV export.
    """
    answer = get_answer(original_query, docs, config)
    relevance = score_relevance(original_query, docs, config)
    faithfulness = score_faithfulness(answer, docs, config)
    completeness = score_completeness(original_query, answer, config)
    overall = round(
        (relevance + faithfulness + completeness) / 3, 4
    )

    return {
        "transformer": transformer_name,
        "original_query": original_query,
        "transformed_query": " | ".join(transformed_queries),
        "chunks_retrieved": len(docs),
        "answer": answer,
        "relevance_score": relevance,
        "faithfulness_score": faithfulness,
        "completeness_score": completeness,
        "overall_score": overall
    }