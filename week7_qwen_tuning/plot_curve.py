"""Draw the frozen test performance curve as a portable SVG."""
import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'
BUDGETS = (2048, 4000, 6000, 8000)
MODES = (
    ('child', 'Child', '#205a95'),
    ('parent', 'Direct parent', '#c65b3e'),
    ('child_to_parent', 'Child to parent', '#9362a6'),
    ('child_to_selected_siblings', 'Selected siblings', '#2a8a65'),
)


def make_plot(rows):
    data = {(r['mode'], int(r['context_budget'])): r for r in rows if int(r['k']) == 10}
    if len(data) != len(MODES) * len(BUDGETS):
        raise ValueError('Expected one frozen Hit@10 row for each mode and budget')
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="520" viewBox="0 0 1180 520">',
             '<rect width="1180" height="520" fill="#ffffff"/>',
             '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263442} .title{font-size:21px;font-weight:700} '
             '.sub{font-size:12px;fill:#65717b} .label{font-size:14px;font-weight:700} '
             '.tick{font-size:12px;fill:#586875} .legend{font-size:13px}</style>',
             '<text x="46" y="40" class="title">Qwen retrieval across context budgets</text>',
             '<text x="46" y="62" class="sub">Weights selected on 20 development questions; frozen evaluation on 80 test questions</text>']
    for panel, (title, field, ceiling, ticks, suffix) in enumerate((
        ('Complete gold-child hits at K=10', 'hit_count', 60, (0, 15, 30, 45, 60), '/80'),
        ('Gold-text coverage per 1,000 tokens', 'mean_coverage_efficiency', .00032,
         (0, .00008, .00016, .00024, .00032), ''),
    )):
        x0 = 66 + panel * 567
        x1 = x0 + 455
        y0, y1 = 112, 382
        parts.append(f'<text x="{x0}" y="94" class="label">{escape(title)}</text>')
        for tick in ticks:
            y = y1 - tick / ceiling * (y1-y0)
            label = f'{tick:.0f}{suffix}' if field == 'hit_count' else f'{tick*1000:.2f}'
            parts += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e4eaed"/>',
                      f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end" class="tick">{label}</text>']
        for i, budget in enumerate(BUDGETS):
            x = x0 + i * (x1-x0)/3
            parts += [f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y1+5}" stroke="#8b969d"/>',
                      f'<text x="{x:.1f}" y="{y1+23}" text-anchor="middle" class="tick">{budget:,}</text>']
        parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+47}" text-anchor="middle" class="tick">Context budget (tokens)</text>')
        for mode, label, color in MODES:
            points = []
            for i, budget in enumerate(BUDGETS):
                row = data[(mode, budget)]
                value = float(row[field])
                if field == 'mean_coverage_efficiency':
                    value = value
                x = x0 + i * (x1-x0)/3
                y = y1 - value / ceiling * (y1-y0)
                points.append((x, y))
            coords = ' '.join(f'{x:.1f},{y:.1f}' for x, y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="3"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
    for i, (_, label, color) in enumerate(MODES):
        x = 70 + i * 275
        parts += [f'<line x1="{x}" y1="477" x2="{x+30}" y2="477" stroke="{color}" stroke-width="3"/>',
                  f'<circle cx="{x+15}" cy="477" r="4" fill="{color}"/>',
                  f'<text x="{x+39}" y="482" class="legend">{escape(label)}</text>']
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=RESULTS)
    args = parser.parse_args()
    with (args.results / 'test_frozen.csv').open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    (args.results / 'context_budget_curve.svg').write_text(make_plot(rows), encoding='utf-8')
