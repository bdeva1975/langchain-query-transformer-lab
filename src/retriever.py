# src/retriever.py
# Handles document loading, indexing, and retrieval.

import logging
from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.getLogger("chromadb").setLevel(logging.ERROR)


def get_embeddings(config: dict) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config["embeddings"]["model"]
    )


def load_documents(config: dict) -> list:
    """Load all supported documents from the data directory."""
    input_dir = config["data"]["input_dir"]
    file_types = config["data"]["file_types"]
    all_docs = []

    files = list(Path(input_dir).iterdir())
    supported = [f for f in files if f.suffix.lower() in file_types]

    if not supported:
        raise ValueError(
            f"No supported documents found in '{input_dir}'.\n"
            f"Supported types: {file_types}\n"
            f"Please add at least one PDF or TXT file to data/ folder."
        )

    for file_path in supported:
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"⚠️  Could not load {file_path.name}: {e}")

    return all_docs


def build_index(config: dict) -> Chroma:
    """Load documents, split, embed and store in ChromaDB."""
    docs = load_documents(config)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunking"]["chunk_size"],
        chunk_overlap=config["chunking"]["chunk_overlap"]
    )
    chunks = splitter.split_documents(docs)

    collection = Chroma(
        collection_name=config["chroma"]["collection_name"],
        embedding_function=get_embeddings(config),
        persist_directory=config["chroma"]["persist_dir"]
    )
    collection.reset_collection()
    collection.add_documents(chunks)
    return collection


def get_retriever(collection: Chroma, config: dict):
    """Return a retriever from the ChromaDB collection."""
    return collection.as_retriever(
        search_kwargs={"k": config["retrieval"]["top_k"]}
    )


def retrieve_with_query(
    retrieval_query: str,
    collection: Chroma,
    config: dict
) -> list:
    """
    Retrieve documents using a single query string.
    Used by all transformers except multi-query.
    """
    retriever = get_retriever(collection, config)
    return retriever.invoke(retrieval_query)


def retrieve_multi_query(
    queries: list,
    collection: Chroma,
    config: dict
) -> list:
    """
    Retrieve documents using multiple query variations.
    Deduplicates results by page content.
    Used by the Multi-Query transformer.
    """
    retriever = get_retriever(collection, config)
    seen = set()
    all_docs = []

    for query in queries:
        docs = retriever.invoke(query)
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    return all_docs


def retrieve(
    transformer_result: dict,
    collection: Chroma,
    config: dict
) -> list:
    """
    Smart retrieval dispatcher.
    Routes to single or multi-query retrieval based on transformer.
    """
    transformer = transformer_result["transformer"]
    queries = transformer_result["transformed_queries"]

    if transformer == "Multi-Query Generation":
        return retrieve_multi_query(queries, collection, config)
    else:
        return retrieve_with_query(
            transformer_result["retrieval_query"],
            collection,
            config
        )