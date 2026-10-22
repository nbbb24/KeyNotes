# Machine learning: from a question to a reliable system

Machine learning estimates patterns from data and uses them to make predictions or decisions. Learning an algorithm is only one part: you must define the task, prevent information leakage, select an objective, evaluate the right behavior, and understand deployment constraints.

## Four connected tracks

| Track | Starting point | Practical outcome |
| --- | --- | --- |
| Foundations | [Machine-learning engineering](mle/index.md) | Explain a baseline, its assumptions, and its errors |
| Language and multimodal models | [LLM learning map](llm/index.md) | Connect architectures, objectives, adaptation, and generation |
| Relational data | [SQL](sql.md) | Build a dataset with the intended keys and row meaning |
| Data analysis | [Pandas](pandas.md) | Inspect, transform, and validate data reproducibly |

## A beginner's route

Choose a small prediction task. Define the target and what information is available at prediction time. Inspect missingness and duplicate entities, split data appropriately, and fit a simple model. Evaluate it using a metric aligned with the decision. Explain at least three errors before selecting a more complex algorithm.

The foundational route covers [learning paradigms](mle/learning-paradigms.md), [data cleaning](mle/data-cleaning.md), [regression](mle/linear-logistic.md), [losses](mle/loss-functions.md), and [metrics](mle/metrics.md). You can learn probability and linear algebra alongside concrete examples rather than treating all mathematics as an entrance exam.

## An advanced route

Ask how conclusions change when assumptions fail: correlated samples, changing populations, missing labels, limited context, expensive inference, or imperfect retrieval. Connect [experiment design](mle/improving-networks.md) and [training diagnosis](mle/training-challenges.md) to each model decision. Derive an objective, implement a small example, and compare its behavior with the full system.

For language systems, evaluate retrieval and generation separately, distinguish factual grounding from fluent output, and measure memory and latency alongside quality. [Infra](../infra/index.md) explains why model improvements must coexist with finite bandwidth, memory, and synchronization costs.

## A project that joins the tracks

Build a small document question-answering evaluation. Use SQL or Pandas to verify source identities and splits. Establish a simple retrieval baseline, add [RAG](llm/rag.md), and examine failures by retrieval coverage, evidence use, and answer correctness. Measure token count, latency, and memory. Only then decide whether better data, a different retriever, fine-tuning, or a systems change addresses the bottleneck.

Each topic includes a mechanism, assumptions, and examples so you can return at a deeper level. The aim is to explain why a method works and recognize when it does not.
