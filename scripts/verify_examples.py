"""Run explicitly marked, self-contained educational Python examples."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    count = 0
    failures = []
    for page in sorted((ROOT / 'docs/en').rglob('*.md')):
        for number, code in enumerate(re.findall(r'```python\n(.*?)\n```', page.read_text(), re.S)):
            if '# runnable-example' not in code:
                continue
            count += 1
            result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=20)
            if result.returncode:
                failures.append(f'{page.relative_to(ROOT)} block {number}:\n{result.stderr}')
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'PASS: {count} executable English examples.')


if __name__ == '__main__':
    main()
