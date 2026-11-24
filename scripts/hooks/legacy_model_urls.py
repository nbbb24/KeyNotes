"""Keep published model URLs working after moving references into the LLM section."""
from html import escape
import json
import os
from pathlib import Path

from mkdocs.plugins import event_priority


MODEL_ROUTES = {
    'index': 'ml/llm/model-structures',
    'shape-conventions': 'ml/llm/model-structures/shape-conventions',
    'compressed-attention': 'ml/llm/components/compressed-attention',
    'glm-5-1': 'ml/llm/model-structures/glm-5-1',
    'glm-5-2': 'ml/llm/model-structures/glm-5-2',
    'deepseek-v4': 'ml/llm/model-structures/deepseek-v4',
    'deepseek-v4-1': 'ml/llm/model-structures/deepseek-v4-1',
}


@event_priority(-200)
def on_post_build(config):
    # i18n builds all languages at priority -100. Its site_dir remains the root;
    # relative destinations also work under Read the Docs' /en/latest/ prefix.
    site = Path(config.site_dir)
    for locale in ('', 'zh'):
        for slug, destination in MODEL_ROUTES.items():
            old_route = 'models' if slug == 'index' else f'models/{slug}'
            old = site / locale / old_route / 'index.html'
            new = site / locale / destination / 'index.html'
            if not config.use_directory_urls:
                if slug != 'index':
                    old = site / locale / f'models/{slug}.html'
                if destination != 'ml/llm/model-structures':
                    new = site / locale / f'{destination}.html'
            target = Path(os.path.relpath(new, old.parent)).as_posix()
            if config.use_directory_urls:
                target = target.removesuffix('index.html')
            language = locale or 'en'
            label = '页面已移动，前往新位置。' if locale == 'zh' else 'This page has moved. Continue to its new location.'
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_text(
                f'<!doctype html><html lang="{language}"><head><meta charset="utf-8">'
                f'<title>Page moved</title><meta name="robots" content="noindex">'
                f'<link rel="canonical" href="{escape(target, quote=True)}">'
                f'<meta http-equiv="refresh" content="0; url={escape(target, quote=True)}">'
                '</head><body>'
                f'<p><a href="{escape(target, quote=True)}">{label}</a></p>'
                f'<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>'
                '</body></html>\n', encoding='utf-8',
            )
