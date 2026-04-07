# 🔬 LangChain Query Transformer Lab

> **Compare 4 RAG query transformation techniques side by side — with a live Streamlit UI.**

Most RAG systems fail not because of bad documents, but because of **bad queries**. This lab shows you exactly how each transformation technique changes what gets retrieved — and which one works best for your data.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.25-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-red)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0.9-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 What It Does

Type a query, select your transformers, and instantly see:
- How each technique **rewrites or expands** your original query
- What chunks each transformer **retrieves** from your documents
- The **generated answer** from each transformer
- A **scored leaderboard** comparing Relevance, Faithfulness and Completeness

```
🏆 LEADERBOARD
───────────────────────────────────────────────────────────────
│ Rank │ Transformer              │ Relevance │ Faithfulness │ Completeness │ Overall │
│    1 │ HyDE                     │    0.6567 │       1.0000 │       1.0000 │  0.8856 │
│    2 │ Baseline                 │    0.6461 │       1.0000 │       1.0000 │  0.8820 │
│    3 │ Step-Back Questioning    │    0.5999 │       1.0000 │       1.0000 │  0.8666 │
│    4 │ Rewrite-Retrieve-Read    │    0.6278 │       0.8000 │       1.0000 │  0.8093 │
│    5 │ Multi-Query Generation   │    0.5916 │       0.0000 │       1.0000 │  0.5305 │
───────────────────────────────────────────────────────────────
🥇 Best transformer: HyDE — Overall score: 0.8856
```

---

## 🔬 The 4 Transformers

| # | Transformer | How It Works | Best For |
|---|-------------|--------------|----------|
| 1 | **Baseline** | No transformation — original query used as-is | Reference comparison |
| 2 | **Rewrite-Retrieve-Read** | LLM rewrites query into clearer, more precise form | Poorly worded queries |
| 3 | **Multi-Query Generation** | Generates N query variations for broader retrieval | Abstract or broad queries |
| 4 | **Step-Back Questioning** | Generates a higher-level abstract question | Specific queries needing context |
| 5 | **HyDE** | Generates a hypothetical answer and uses it as query | Vocabulary mismatch between query and docs |

---

## 📊 Scoring Dimensions

Each transformer is scored on 3 dimensions:

- **Relevance** — Cosine similarity between original query embedding and retrieved context embedding
- **Faithfulness** — LLM-judged: is the answer grounded in the retrieved context?
- **Completeness** — LLM-judged: does the answer fully address the question?
- **Overall** — Average of the three scores above

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/bdeva1975/langchain-query-transformer-lab.git
cd langchain-query-transformer-lab
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key
Create a `.env` file in the root folder:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 5. Add your documents
Drop any `.pdf` or `.txt` files into the `data/` folder.

### 6. Run the app
```bash
streamlit run app.py
```

### 7. In the UI
1. Click **🔄 Build / Rebuild Index** in the sidebar
2. Type your query in the main area
3. Select which transformers to run
4. Click **🚀 Run Transformers**
5. Compare results across tabs and the leaderboard

---

## ⚙️ Configuration

All settings are in `config.yaml`:

```yaml
llm:
  model: "gpt-4o-mini"        # change to gpt-4o for higher quality
  temperature: 0
  max_tokens: 500

embeddings:
  model: "text-embedding-3-small"

retrieval:
  top_k: 3                    # chunks retrieved per query

transformers:
  multi_query:
    num_queries: 3            # number of query variations to generate
```

---

## 📁 Project Structure

```
langchain-query-transformer-lab/
│
├── src/
│   ├── transformers.py   # All 4 query transformation techniques
│   ├── retriever.py      # Document loading, indexing, retrieval
│   └── evaluator.py      # Scoring: relevance, faithfulness, completeness
│
├── data/                 # ← Put your documents here
│
├── app.py                # Streamlit UI — main entry point
├── config.yaml           # All settings
└── requirements.txt      # Dependencies
```

---

## 💡 Key Insights

**When does each transformer shine?**

- **Rewrite** works best when users type vague or grammatically poor queries
- **Multi-Query** works best with large knowledge bases where one query might miss relevant sections
- **Step-Back** works best when the query is too specific and misses broader context
- **HyDE** works best when query vocabulary differs from document vocabulary — common in technical or scientific domains

**Performance varies by document type.** Always benchmark on your own data before choosing a transformer for production.

---

## 🔗 Related Projects

- [rag-indexing-benchmark](https://github.com/bdeva1975/rag-indexing-benchmark) — Compare 6 RAG indexing strategies on your own documents

---

## 📖 Based On

Concepts and techniques from:
> *AI Agents and Applications with LangChain, LangGraph and MCP* — Roberto Infante (Manning, 2026)
> Chapter 9: Question Transformations

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*If this repo helped you, please consider giving it a ⭐ — it helps others find it.*
