# learning-hub-ai

AI Services for the Multimodal AI Learning Hub. Handles all LLM processing, LangGraph workflows, and RAG pipeline.

## Overview

This repository contains the AI/LLM services that power the intelligent features of the learning hub:
- Intent classification
- Semantic search (RAG)
- Response generation
- Quiz generation
- Essay grading
- Flashcard generation

## Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | LangGraph |
| LLM (Generation) | Gemini 1.5 Pro |
| LLM (Fast) | Groq (Llama 3) |
| Vector DB | Qdrant |
| API | FastAPI |

## Directory Structure

```
learning-hub-ai/
├── src/
│   ├── agents/              # AI agents (classifier, retriever, etc.)
│   │   ├── classifier.py
│   │   ├── retriever.py
│   │   ├── grader.py
│   │   ├── generator.py
│   │   └── reflection.py
│   ├── llm/               # LLM clients
│   │   ├── gemini.py
│   │   └── groq.py
│   ├── rag/               # RAG pipeline
│   │   ├── pipeline.py
│   │   └── search.py
│   ├── workflows/         # LangGraph workflows
│   │   ├── qa.py
│   │   ├── quiz.py
│   │   └── essay.py
│   └── prompts/          # System prompts
├── tests/
├── docs/                  # This documentation
├── Dockerfile
├── requirements.txt
└── main.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn main:app --reload --port 8001
```

## Environment Variables

```env
# Required
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Internal API (for worker calls)
INTERNAL_API_KEY=your_internal_key
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ai/v1/classify` | Classify user intent |
| POST | `/ai/v1/query` | RAG query (non-streaming) |
| POST | `/ai/v1/query/stream` | RAG query (streaming) |
| POST | `/ai/v1/quiz/generate` | Generate quiz |
| POST | `/ai/v1/essay/grade` | Grade essay |
| GET | `/health` | Health check |

## Workflows

### Q&A Workflow

```
User Query → Intent Classifier → Retriever → Grader → Generator → Reflection → Response
```

### Quiz Generation

```
Request → Intent Classifier → Retriever → Quiz Generator → Questions
```

### Essay Grading

```
Essay → Intent Classifier → Retriever → Essay Grader → Score + Feedback
```

## Related Documentation

- [Main Docs](../README.md) - System overview
- [API Contracts](../communication/api-contracts.md) - Service contracts
- [System Design](../3-architecture/system-design.md) - Architecture details