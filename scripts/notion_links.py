"""Render Notion export URLs without changing the exported Markdown files."""

from pathlib import Path
import re
from urllib.parse import unquote

LOCAL_LINK = re.compile(r"(?<=\]\()([^\n]+?\.(?:md|png))(?=\))")


def on_page_markdown(markdown, page, config, files):
    if not page.file.src_uri.startswith("ml/"):
        return markdown
    parent = Path(page.file.abs_src_path).parent

    def encode_parentheses(match):
        target = match.group(0)
        if not (parent / unquote(target)).is_file():
            return target
        return target.replace("(", "%28").replace(")", "%29")

    return LOCAL_LINK.sub(encode_parentheses, markdown)
