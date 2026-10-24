# 🔗 LangChain: compose a model application around explicit state

LangChain provides abstractions for model applications, integrations, tools, and agent workflows. Its API evolves, so pin package versions and consult the documentation for the installed release. The enduring lesson is to make data flow and side effects explicit.

## 🏗️ Separate the components

A model predicts messages or tool calls. A retriever returns candidate evidence. A tool executes a defined operation. An orchestrator decides how outputs feed subsequent steps and when execution stops. Conversation state records what the workflow needs to continue; it is not a substitute for a durable source of truth.

A simple retrieval application can be expressed without an autonomous agent:

```text
question -> retrieve passages -> assemble evidence -> generate answer -> validate citations
```

An agent loop adds a model-driven choice of the next action. This flexibility is useful when the route cannot be fixed in advance, but it also requires iteration limits, error handling, and visibility into intermediate actions.

## 🔄 A practical support workflow

For a support assistant, begin with a deterministic document-search path. Add a tool for looking up a ticket only if the task requires it. Keep the ticket identifier, retrieved sources, and tool outcome in structured state. If a write is needed, represent it as a distinct operation with clear authorization and retry behavior.

Current LangChain documentation presents `create_agent` as a configurable agent interface. LangGraph provides lower-level workflow orchestration, and LangSmith supports tracing/evaluation. Choose the level of abstraction that makes the application's control flow understandable; adopting every component is unnecessary.

## 📊 Evaluate behavior rather than framework usage

Record retrieval quality, tool-call validity, task success, latency, cost, and failure recovery. A trace should show whether an error originated in retrieval, model selection, tool execution, or final synthesis. Avoid logging sensitive content merely because tracing is convenient.

Provider abstractions do not make all models identical. Message formats, streaming events, structured outputs, and tool schemas can differ. Test the actual combination of provider, model, package version, and middleware.

**Exercise.** Does wrapping a function as a tool make it safe to retry? No. Idempotency depends on what the function does. A search and a ticket creation require different retry policies.

Reference: [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview). Related: [MCP](mcp.md), [RAG](../rag.md).
