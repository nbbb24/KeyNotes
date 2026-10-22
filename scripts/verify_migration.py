"""Verify topic coverage, original image bytes, local links, and language alternates."""
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site'


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.images = []
        self.alternates = {}
        self.language = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'html':
            self.language = attrs.get('lang')
        if tag == 'a' and 'href' in attrs:
            self.links.append(attrs['href'])
            if 'hreflang' in attrs:
                self.alternates[attrs['hreflang']] = attrs['href']
        if tag == 'img' and 'src' in attrs:
            self.images.append(attrs['src'])


def target(page, url):
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or not parts.path:
        return None
    decoded = unquote(parts.path)
    result = SITE / decoded.lstrip('/') if decoded.startswith('/') else page.parent / decoded
    result = result.resolve()
    return result / 'index.html' if result.is_dir() else result


def output_path(topic, language):
    path = Path(topic['target'])
    html = path.with_suffix('.html') if path.name == 'index.md' else path.with_suffix('') / 'index.html'
    return SITE / ('zh' if language == 'zh' else '') / html


def main():
    errors = []
    topics = json.loads((ROOT / 'migration/topic-map.json').read_text())
    images = json.loads((ROOT / 'migration/image-manifest.json').read_text())
    expected = set()
    for item in images:
        source = ROOT / item['path']
        if not source.is_file() or sha256(source.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f"Original image changed: {source}")
        built = SITE / source.relative_to(ROOT / 'docs')
        expected.add(built.resolve())
        if not built.is_file() or sha256(built.read_bytes()).hexdigest() != item['sha256']:
            errors.append(f"Built image changed or missing: {built}")
    rendered = set()
    for page in SITE.rglob('*.html'):
        parser = References()
        parser.feed(page.read_text())
        for url in parser.links + parser.images:
            resolved = target(page, url)
            if resolved is not None and not resolved.exists():
                errors.append(f"Broken reference: {page.relative_to(SITE)} -> {url}")
        rendered.update(p for u in parser.images if (p := target(page, u)) is not None)
    for path in expected - rendered:
        errors.append(f"Original image never rendered: {path}")
    for item in topics:
        for lang in ('en', 'zh'):
            source = ROOT / 'docs' / lang / item['target']
            page = output_path(item, lang)
            if not source.is_file() or not page.is_file():
                errors.append(f"Missing topic: {lang}/{item['target']}")
                continue
            raw = source.read_text()
            if '{{link:' in raw or '{{figure' in raw:
                errors.append(f"Unresolved editorial placeholder: {source}")
            parser = References()
            parser.feed(page.read_text())
            if parser.language != lang:
                errors.append(f"Incorrect page language: {page}")
            other = 'zh' if lang == 'en' else 'en'
            link = parser.alternates.get(other)
            if not link or target(page, link) != output_path(item, other).resolve():
                errors.append(f"Incorrect same-topic language switch: {page}")
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"PASS: {len(topics)} topics in each language; {len(images)} original images unchanged; "
          'local links and same-topic language switches valid.')


if __name__ == '__main__':
    main()
