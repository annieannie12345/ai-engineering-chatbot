# AI Engineering Concepts Guide

This starter knowledge base is for local development and portfolio demos. Expand it with your own notes, course material, architecture decisions, research paper summaries, documentation, and project writeups.

## What is machine learning?

Machine learning is a way to build software that learns patterns from data instead of being programmed with every rule manually. A model receives examples, learns statistical relationships, and then makes predictions or decisions on new inputs.

Common machine learning tasks include classification, regression, clustering, recommendation, anomaly detection, ranking, and forecasting. A typical workflow includes collecting data, cleaning it, selecting features, training a model, validating it, testing it, deploying it, and monitoring performance after release.

## Supervised, unsupervised, and reinforcement learning

Supervised learning uses labeled examples, such as images labeled as cats or dogs. The model learns to map inputs to known outputs.

Unsupervised learning works with unlabeled data. It is often used to discover structure, such as clusters, topics, or lower-dimensional representations.

Reinforcement learning trains an agent to choose actions in an environment. The agent receives rewards or penalties and learns a policy that maximizes long-term reward.

## What is deep learning?

Deep learning is a subfield of machine learning that uses neural networks with many layers. These networks can learn useful representations from raw data such as text, images, audio, and time series.

Neural networks are built from layers of parameters. During training, the model makes predictions, measures error with a loss function, and updates parameters using backpropagation and an optimizer such as stochastic gradient descent or Adam.

## What are embeddings?

Embeddings are numerical vectors that represent the meaning or properties of data. In language systems, text with similar meaning should have embeddings that are close together in vector space.

Embeddings are useful for semantic search, clustering, recommendations, duplicate detection, retrieval, and RAG systems. An embedding model converts text into vectors, and a vector database stores those vectors for fast similarity search.

## What is a vector database?

A vector database stores embeddings and retrieves items that are close to a query embedding. It is commonly used when keyword search is not enough and the system needs semantic similarity.

In a RAG pipeline, documents are split into chunks, embedded, and stored in a vector database. At question time, the question is embedded, similar chunks are retrieved, and those chunks are sent to the language model as context.

## What is Retrieval-Augmented Generation?

Retrieval-Augmented Generation, or RAG, combines search with generation. Instead of asking a language model to answer only from its internal training data, the application retrieves relevant context from a knowledge base and includes it in the prompt.

RAG is useful when answers need to be grounded in private, recent, or domain-specific documents. A basic RAG pipeline includes document loading, chunking, embedding, vector storage, retrieval, prompt construction, generation, and citation or source display.

## RAG design choices

Chunk size controls how much text is stored in each retrievable unit. Small chunks can improve precision but may lose context. Large chunks preserve context but can retrieve irrelevant text.

Chunk overlap helps preserve meaning across chunk boundaries. Top-k controls how many chunks are retrieved. Reranking can improve quality by reordering retrieved chunks before generation. Metadata filters can restrict retrieval to a project, file, date, or topic.

## What is a large language model?

A large language model, or LLM, is a neural network trained to predict and generate text. Modern LLMs can summarize, answer questions, write code, reason over provided context, and follow instructions.

LLMs are not databases. They can produce plausible but incorrect answers, especially when asked about facts not present in the prompt or training data. Production systems use grounding, retrieval, evaluation, guardrails, logging, and human review to reduce risk.

## Prompt engineering

Prompt engineering is the practice of giving a model clear instructions, context, examples, constraints, and output formats. Good prompts reduce ambiguity and make the desired behavior easier for the model to follow.

Useful prompt patterns include role instructions, task instructions, context blocks, examples, step-by-step decomposition, structured output schemas, and refusal criteria for unsupported questions.

## Fine-tuning vs RAG

RAG is best when the model needs access to external facts, private documents, or frequently changing knowledge. Fine-tuning is best when the model needs to learn a style, format, domain behavior, or repeated task pattern.

Fine-tuning does not reliably turn a model into a knowledge database. Many systems use both: RAG for factual grounding and fine-tuning for behavior or formatting.

## AI agents

An AI agent is a system that uses a model to decide actions, call tools, observe results, and continue working toward a goal. Agents can search, write files, query APIs, run code, and coordinate multi-step workflows.

Good agent design requires clear tool boundaries, careful permissions, state tracking, error handling, and evaluation. Agents are powerful but can be unreliable if tasks are vague or tools are unsafe.

## Evaluation

AI evaluation measures whether a system behaves correctly. For RAG systems, common checks include retrieval relevance, answer faithfulness, citation accuracy, refusal behavior, latency, and cost.

Evaluation can include golden question-answer sets, human review, automated model-based grading, regression tests, and production monitoring. Strong evaluation turns an AI demo into an engineering system.

## MLOps and LLMOps

MLOps covers practices for taking machine learning systems from experiment to production. It includes data versioning, training pipelines, model registries, deployment, monitoring, rollback, and governance.

LLMOps extends these practices to language model applications. Important concerns include prompt versioning, retrieval quality, token usage, model latency, safety checks, observability, and continuous evaluation.

## Building an AI engineering project

A strong AI engineering project should have a clear problem statement, reliable data flow, modular code, configuration through environment variables, tests, documentation, and a repeatable setup process.

For a RAG chatbot, the main components are a frontend, backend API, document ingestion pipeline, embedding model, vector database, retriever, prompt template, LLM, source display, and evaluation workflow.
