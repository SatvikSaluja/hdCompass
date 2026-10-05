"""Build the GitHub Pages snapshot. Run with .venv/bin/python (markdown-it-py required)."""
from pathlib import Path
from html import escape
import json
import re
import shutil
import zipfile
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'
OUT.mkdir(exist_ok=True)
source = (ROOT / 'results.html').read_text()
style = re.search(r'<style>(.*?)</style>', source, re.S).group(1)
style += '\nimg{max-width:100%;height:auto}pre{overflow:auto;padding:16px;background:#edf0ed}td{white-space:normal;min-width:80px}.scroll td{white-space:nowrap}code{overflow-wrap:anywhere}h2{margin-top:28px}'

def page(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)} — hdcompass</title><style>{style}</style></head><body><main><header><div class="meta">HDCOMPASS · RESULTS</div><h1>{escape(title)}</h1><nav aria-label="Page sections"><a href="index.html">Overview</a><a href="report.html">Full report</a><a href="archive.html">All reports and figures</a><a href="results.zip">Download all results</a></nav></header>{body}<footer>Static results snapshot · No external dependencies</footer></main></body></html>'

md = MarkdownIt('commonmark', {'html': False}).enable('table')
for filename, output, title in [('RESULTS.md','report.html','Full results and validation'),('README.md','guide.html','Project and run guide')]:
    text = (ROOT / filename).read_text()
    html = md.render(text).replace('href="RESULTS.md"', 'href="report.html"')
    html = html.replace('<table>', '<div class="scroll"><table>').replace('</table>', '</table></div>')
    (OUT / output).write_text(page(title, html))
    shutil.copy2(ROOT / filename, OUT / filename)

files = []
for folder in ['results', 'figures']:
    for path in sorted((ROOT / folder).rglob('*')):
        if not path.is_file() or path.suffix not in {'.json','.png','.log'}:
            continue
        if path.suffix == '.json':
            json.loads(path.read_text())
        relative = path.relative_to(ROOT)
        dest = OUT / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files.append(relative)
for filename in ['validate.py','run_dandi000939.py']:
    (OUT / 'scripts').mkdir(exist_ok=True)
    shutil.copy2(ROOT / 'scripts' / filename, OUT / 'scripts' / filename)

sections = []
for folder in sorted({p.parent for p in files}):
    links = ''.join(f'<li><a href="{p.as_posix()}">{escape(p.name)}</a> <small>({(OUT / p).stat().st_size:,} bytes)</small></li>' for p in files if p.parent == folder)
    sections.append(f'<section><h2>{escape(folder.as_posix())}</h2><ul>{links}</ul></section>')
intro = f'<p>All {len(files)} recorded JSON reports, PNG figures and run logs available in this checkout. Original and frozen runs are listed separately. Files under figures/quick are smoke-test outputs, not full-run evidence.</p><p><a href="RESULTS.md">Download the written report (Markdown)</a> · <a href="manifest.json">File manifest</a></p>'
(OUT / 'archive.html').write_text(page('All reports and figures', intro + ''.join(sections)))
(OUT / 'manifest.json').write_text(json.dumps([{'path': p.as_posix(), 'bytes': (OUT / p).stat().st_size} for p in files], indent=2) + '\n')
with zipfile.ZipFile(OUT / 'results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for p in files + [Path('RESULTS.md'), Path('README.md'), Path('manifest.json')]:
        archive.write(OUT / p, p.as_posix())
source = source.replace('href="RESULTS.md"', 'href="report.html"').replace('href="README.md"', 'href="guide.html"')
source = source.replace('<a href="#sources">Source records</a>', '<a href="#sources">Source records</a><a href="report.html">Full report</a><a href="archive.html">All reports and figures</a>')
source = source.replace('<ul><li><a href="report.html">', '<ul><li><a href="archive.html">Browse every report, figure and run log</a></li><li><a href="results.zip">Download all results (ZIP)</a></li><li><a href="report.html">')
(OUT / 'index.html').write_text(source)
(OUT / 'results.html').write_text(source)
(OUT / '.nojekyll').touch()
print(f'Built site with {len(files)} result artifacts in {OUT}')
