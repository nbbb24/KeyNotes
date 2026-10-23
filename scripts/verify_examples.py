"""Run explicitly marked, self-contained educational Python examples."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    counts = {'en': 0, 'zh': 0}
    failures = []
    for language in counts:
        for page in sorted((ROOT / 'docs' / language).rglob('*.md')):
            blocks = re.findall(r'```python\n(.*?)\n```', page.read_text(), re.S)
            for number, code in enumerate(blocks):
                if '# runnable-example' not in code:
                    continue
                counts[language] += 1
                result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
                if result.returncode:
                    failures.append(f'{page.relative_to(ROOT)} block {number}:\n{result.stderr}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {sum(counts.values())} executable examples '
          f'({counts["en"]} English, {counts["zh"]} Chinese).')


if __name__ == '__main__':
    main()
