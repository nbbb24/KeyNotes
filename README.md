# KeyNotes

[![Read the Docs](https://img.shields.io/badge/Docs-Read%20the%20Docs-blue?logo=readthedocs)](https://infra-knowledge.readthedocs.io/en/latest/)
[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)

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

## Sharing and citation

Original KeyNotes documentation text and diagrams are shared under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) where the contributors hold the necessary rights. You may share and adapt them, including commercially, with attribution, a license link, and an indication of changes. Imported images and third-party material are excluded unless their own notices say otherwise; see [LICENSE.md](LICENSE.md) for scope.

Suggested credit: **nbbb24, “KeyNotes,”** [project website](https://infra-knowledge.readthedocs.io/en/latest/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), plus the page URL and any changes you made. GitHub's “Cite this repository” menu also reads [`CITATION.cff`](CITATION.cff).
