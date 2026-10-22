# KeyNotes

Machine learning and infrastructure, explained through mechanisms, worked examples, and diagrams.

- **[English](docs/en/index.md)** is the default website language.
- **[简体中文](docs/zh/index.md)** is available through the top-right language icon, which keeps the current topic.
- **Infra** and **ML** are sibling sections. ML includes LLMs, multimodal learning, MLE foundations, SQL, and Pandas.

The English editorial edition covers 113 pages. The Chinese edition currently retains the migrated notes and is being rewritten to match the English edition; unfinished pages display an explicit notice.

## Preview and verify

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/mkdocs.txt
mkdocs serve --dev-addr 127.0.0.1:8000
```

Open <http://127.0.0.1:8000/>. Chinese pages are under `/zh/`. On Windows, activate with `.venv\Scripts\Activate.ps1`.

```bash
mkdocs build --strict
python scripts/verify_migration.py
python -m pip install pandas
python scripts/verify_examples.py
```

`site/` is generated output and is not committed. Formula rendering uses MathJax; new explanatory diagrams use Mermaid. Original figures are shared by both languages without recompression or redrawing.

## Read the Docs

Import `nbbb24/KeyNotes` into Read the Docs and select `main`. The repository includes `.readthedocs.yaml` and its MkDocs dependencies. Both languages are built as one site, with English at the root and Chinese under `zh/`; a separate RTD translation project is unnecessary. A successful local build does not by itself provision an RTD project. Once the project and webhook are connected, pushes can trigger builds.

## Editing

Keep corresponding pages at identical relative paths in `docs/en/` and `docs/zh/`. Add both to the topic map and update navigation in `mkdocs.yml`. Explain motivation, mechanism, assumptions, a concrete example, and failure modes as appropriate. Link primary references for research or implementation-specific claims. Avoid treating a model family name as a fixed specification or a benchmark score as a universal ranking.

Store shared images in `docs/assets/`. Original screenshots can contain outdated labels or mistakes; retain their bytes and explain the correction in adjacent prose. Preserve meaningful equations, examples, and figure explanations in both languages.

## Migration provenance

The original Notion ML export contained 104 Markdown pages and 215 PNG images. ECGBYTE was excluded at the owner's request, leaving 103 ML pages. Together with nine Infra pages and the homepage, this gives 113 topics per language.

The initial verbatim migration is preserved in Git commit `614e5ae`. The later editorial rewrite was explicitly requested by the owner. These records distinguish original material from edited explanations:

- `migration/notion-ml-manifest.json` and `migration/infra-manifest.json`: immutable records of the initial migration, not hashes of the rewritten prose.
- `migration/topic-map.json`: original topic paths mapped to readable current paths.
- `migration/image-manifest.json`: original-to-current image paths and SHA-256 checksums for all 215 PNG and two Infra SVG files.

The verification script checks both language trees, original image bytes, rendered image references, local links, and same-topic language-switch destinations. The original export's filenames are preserved in provenance records; the live pages use readable URLs.
