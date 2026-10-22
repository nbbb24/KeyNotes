"""Verify original content, built image bytes, and local site links after mkdocs build."""

from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "a" and "href" in attrs:
            self.links.append(attrs["href"])
        if tag == "img" and "src" in attrs:
            self.images.append(attrs["src"])


def local_target(page, url):
    path = urlsplit(url)
    if path.scheme or path.netloc or not path.path:
        return None
    decoded = unquote(path.path)
    target = SITE / decoded.lstrip("/") if decoded.startswith("/") else page.parent / decoded
    target = target.resolve()
    return target / "index.html" if target.is_dir() else target


def main():
    errors = []
    source_count = 0
    expected_images = set()
    for name in ("infra-manifest.json", "notion-ml-manifest.json"):
        manifest = json.loads((ROOT / "migration" / name).read_text())
        for item in manifest["files"]:
            source = ROOT / item["path"]
            source_count += 1
            if not source.is_file() or sha256(source.read_bytes()).hexdigest() != item["sha256"]:
                errors.append(f"Source differs from migration manifest: {item['path']}")
            if source.suffix == ".png":
                built = SITE / source.relative_to(ROOT / "docs/zh")
                expected_images.add(built.resolve())
                if not built.is_file() or sha256(built.read_bytes()).hexdigest() != item["sha256"]:
                    errors.append(f"Built image differs from original: {built}")

    rendered_images = set()
    pages = list(SITE.rglob("*.html"))
    if not pages:
        errors.append("No built site found; run mkdocs build --strict first.")
    for page in pages:
        parser = References()
        parser.feed(page.read_text())
        for url in parser.links + parser.images:
            target = local_target(page, url)
            if target is not None and not target.exists():
                errors.append(f"Broken local reference in {page.relative_to(SITE)}: {url}")
        for url in parser.images:
            target = local_target(page, url)
            if target is not None:
                rendered_images.add(target)
    for image in expected_images - rendered_images:
        errors.append(f"Image has no rendered reference: {image.relative_to(SITE)}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {source_count} files match migration manifests; {len(expected_images)} images unchanged and rendered; "
          f"{len(pages)} HTML pages; no broken local references.")


if __name__ == "__main__":
    main()
