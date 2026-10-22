# LLM methods and applications: course route

This track turns the original lecture notes into a connected learning sequence. The topical pages retain their diagrams, while the standalone chapters provide deeper derivations and implementation examples.

## Study in dependency order

1. [Introduction](introduction.md): tokens, vectors, probabilities, losses, and decoding.
2. [Architectures](architectures.md): recurrent attention, Transformer blocks, and visibility patterns.
3. [Pretraining](pretraining.md): objectives, tokenization, data quality, and packing.
4. [Architecture advances](architecture-advances.md): gated FFNs, GQA/MQA/MLA, normalization, and positions.
5. [Evaluation](evaluation.md): predictive fit, overlap metrics, semantic metrics, and task checks.
6. [Fine-tuning](finetuning.md): compare full tuning and parameter-efficient methods.
7. [Retrieval research and lab](rag.md): identify where external memory enters a system.

## Build one cumulative project

Use a small document collection and a fixed set of questions. First establish a generation baseline. Add retrieval and inspect evidence. Then test one adaptation method only if error analysis identifies a behavior that training could improve. Keep the test set untouched while changing prompts, retrievers, or training hyperparameters.

For every experiment, record model/version, preprocessing, context budget, decoding, latency, and correctness with supporting sources. A strong project explains why a change helped, where it failed, and what resource it consumed. A list of fashionable methods without controlled comparisons does not answer those questions.

## Completion checks

You should be able to draw the Q/K/V sources in each attention block, compute a small cross-entropy and KV-cache estimate, identify loss-mask leakage, and explain why better retrieval metrics may not produce better final answers. Advanced readers should add an ablation and a failure case for each claimed improvement.

Original resource: [CMU Large Language Model course schedule](https://cmu-llms.org/schedule/). The live site changes across course offerings; these chapters preserve the supplied note topics rather than claiming to mirror the current schedule exactly.
