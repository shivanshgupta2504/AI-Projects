# Project Guide and Explanation

- Adaptive RAG is an **advanced Retrieval-Augmented Generation** strategy that intelligently integrates _dynamic query analysis_ with _self-correcting mechanisms_ to optimize response accuracy.
- Adaptive RAG represents the most sophisticated evolution, addressing a fundamental insight: **not all queries are created equal**.
- The research reveals that real-world queries exhibit vastly different complexity levels:
  - **Simple queries**: “Paris is the capital of what?” — Can be answered directly by LLMs
  - **Multi-hop queries**: “When did the people who captured Malakoff come to the region where Philipsburg is located?” — Requires four reasoning steps

## Understanding the Adaptive RAG Workflow

### 1. Query Routing & Classification

The system begins with a trained complexity classifier that analyzes the incoming question. This isn’t just simple keyword matching; it’s a sophisticated assessment that determines:
- Whether the query needs retrieval at all (parametric knowledge sufficient)
- If retrieval is needed, what level of complexity is required
- The optimal strategy ranges from no-retrieval, single-step, to multi-hop approaches

### 2. Dynamic Knowledge Acquisition Strategy

Based on the complexity classification, the system intelligently routes between:
- **Index-based retrieval**: For queries answerable from the existing knowledge base
- **Web search**: For queries requiring fresh information or when local retrieval fails
- **No retrieval**: For queries answerable directly from the model’s parametric knowledge

### 3. Multi-stage Quality Assurance

The system implements a comprehensive evaluation at multiple decision points:
- **Document Relevance Assessment**: Uses confidence scoring to evaluate retrieval quality
- **Hallucination Detection**: Verifies generated answers are grounded in retrieved evidence
- **Answer Quality Evaluation**: Ensures responses adequately address the original question

## Implementation Guide

### Step 1: Define the State Management System & Essential Constants

> Create src/workflow/state.py:

- We start with the foundation of our system — the state management. This is crucial because it defines how information flows through our graph.
- `GraphState` class acts as the central data structure that flows through every node in our graph workflow. The `question` field holds the user's input query, `generation` stores the LLM's response, `web_search` is a boolean flag that determines whether we need to search the web for additional information, and `documents` contains all the retrieved documents from both local and web sources.

By using `TypedDict`, we ensure type safety while maintaining the flexibility needed for our dynamic workflow.

> Create src/workflow/consts.py:
```python
# Workflow node identifiers
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"  
GENERATE = "generate"
WEBSEARCH = "websearch"
```
These constants define the names of our graph nodes and help maintain consistency throughout our codebase. Having centralized constants makes it easier to refactor and reduces the risk of typos when referencing node names in our workflow definitions.

### Step 2: Define the chat model & the embedding model
