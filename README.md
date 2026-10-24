# KeyNotes

[![Read the Docs](https://readthedocs.org/projects/infra-knowledge/badge/?version=latest)](https://infra-knowledge.readthedocs.io/en/latest/)

KeyNotes is a bilingual learning guide to **machine learning** and the **infrastructure behind it**. It connects first principles, worked examples, diagrams, and engineering trade-offs so newcomers can build intuition and experienced readers can trace how systems work.

- **ML:** foundations, language and multimodal models, training, inference, and data tools.
- **Infra:** accelerators, memory, communication, data movement, and performance.

Read the [English documentation](https://infra-knowledge.readthedocs.io/en/latest/) or switch to 简体中文 with the language icon on any page. The two editions share the same topic structure.

## Contributing

Corrections, clearer examples, diagrams, and new topics are welcome. Open an issue to discuss a larger addition, or submit a pull request with a focused change.

1. Keep the English and Chinese versions of a topic at matching paths under `docs/en/` and `docs/zh/`.
2. Explain the motivation, mechanism, assumptions, and a concrete example where useful. Cite primary sources for implementation-specific claims.
3. Update `mkdocs.yml` when adding a topic. Keep shared images in `docs/assets/` and preserve the original imported figures.
4. In the pull request, describe what changed and how you checked the affected pages. Read the Docs provides a preview for pull requests.

The original Notion material and image checksums are recorded in [`migration/`](migration/).
