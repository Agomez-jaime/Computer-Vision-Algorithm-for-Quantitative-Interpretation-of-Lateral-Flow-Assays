"""
validate_runs.py

Validates every run in /runs/, collects mAP50, mAP50-95, Precision, Recall, F1
and saves an interactive HTML comparison dashboard.

Usage:
    python validate_runs.py
"""

from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import json

# ── Paths ─────────────────────────────────────────────────────────
DATASET_DST = "C:\\Users\\Asus\\PyCharmMiscProject\\Covid LFAs\\lighting_conditions\\random variation"
RUNS_DIR    = "C:\\Users\\Asus\\PyCharmMiscProject\\Covid LFAs\\lighting_conditions\\random variation\\runs"
# ─────────────────────────────────────────────────────────────────


def validate_all():
    runs_dir    = Path(RUNS_DIR)
    dataset_dst = Path(DATASET_DST)
    yaml_path   = dataset_dst / "dataset.yaml"

    run_folders = sorted([
        d for d in runs_dir.iterdir()
        if d.is_dir() and (d / "weights" / "best.pt").exists()
    ])

    if not run_folders:
        print("No completed runs found (no weights/best.pt).")
        return

    print(f"Found {len(run_folders)} runs to validate.\n")
    results = []

    for i, run_dir in enumerate(run_folders):
        print(f"[{i+1}/{len(run_folders)}] Validating {run_dir.name} ...")
        model = YOLO(str(run_dir / "weights" / "best.pt"))

        metrics = model.val(
            data=str(yaml_path),
            verbose=False,
            plots=True,
            save_dir=run_dir / "val",
        )

        # Extract mask metrics (segmentation)
        p   = float(metrics.seg.p.mean())   if hasattr(metrics, "seg") else float(metrics.box.p.mean())
        r   = float(metrics.seg.r.mean())   if hasattr(metrics, "seg") else float(metrics.box.r.mean())
        mp50   = float(metrics.seg.map50)   if hasattr(metrics, "seg") else float(metrics.box.map50)
        mp5095 = float(metrics.seg.map)     if hasattr(metrics, "seg") else float(metrics.box.map)
        f1  = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        results.append({
            "run":       run_dir.name,
            "mAP50":     round(mp50,   4),
            "mAP50-95":  round(mp5095, 4),
            "Precision": round(p,      4),
            "Recall":    round(r,      4),
            "F1":        round(f1,     4),
        })
        print(f"   mAP50={mp50:.4f}  P={p:.4f}  R={r:.4f}  F1={f1:.4f}")

    df = pd.DataFrame(results).sort_values("mAP50", ascending=False).reset_index(drop=True)

    # Save CSV
    csv_path = runs_dir / "validation_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved to: {csv_path}")

    # Save interactive HTML dashboard
    html_path = runs_dir / "validation_dashboard.html"
    _save_dashboard(df, html_path)
    print(f"Dashboard saved to: {html_path}")
    print(f"\nBest run: {df.iloc[0]['run']}  (mAP50={df.iloc[0]['mAP50']})")


def _save_dashboard(df: pd.DataFrame, path: Path):
    runs_json    = json.dumps(df["run"].tolist())
    map50_json   = json.dumps(df["mAP50"].tolist())
    map5095_json = json.dumps(df["mAP50-95"].tolist())
    prec_json    = json.dumps(df["Precision"].tolist())
    rec_json     = json.dumps(df["Recall"].tolist())
    f1_json      = json.dumps(df["F1"].tolist())

    # Build table rows
    rows = ""
    for i, row in df.iterrows():
        rank = i + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        rows += f"""
        <tr class="{'best-row' if rank == 1 else ''}">
            <td>{medal}</td>
            <td class="run-name" title="{row['run']}">{row['run']}</td>
            <td>{row['mAP50']:.4f}</td>
            <td>{row['mAP50-95']:.4f}</td>
            <td>{row['Precision']:.4f}</td>
            <td>{row['Recall']:.4f}</td>
            <td>{row['F1']:.4f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>YOLO Run Comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  :root {{
    --bg:      #0d0f14;
    --panel:   #13151c;
    --border:  #1f2330;
    --accent:  #00e5ff;
    --accent2: #ff4d6d;
    --accent3: #a0ff6f;
    --accent4: #ffb347;
    --accent5: #c084fc;
    --text:    #e2e8f0;
    --muted:   #64748b;
  }}

  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    padding: 2rem;
  }}

  h1 {{
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, var(--accent), var(--accent5));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.25rem;
  }}

  .subtitle {{
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    margin-bottom: 2rem;
  }}

  .metric-selector {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1.5rem;
  }}

  .metric-btn {{
    padding: 0.4rem 1rem;
    border-radius: 999px;
    border: 1.5px solid var(--border);
    background: transparent;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    cursor: pointer;
    transition: all 0.2s;
  }}

  .metric-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .metric-btn.active {{ background: var(--accent); border-color: var(--accent); color: #000; font-weight: 700; }}

  .chart-panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    position: relative;
  }}

  .chart-panel canvas {{ max-height: 420px; }}

  .sort-hint {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: var(--muted);
    position: absolute;
    top: 1.2rem;
    right: 1.5rem;
  }}

  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: var(--panel);
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
  }}

  th {{
    padding: 0.75rem 1rem;
    text-align: left;
    background: #0a0c10;
    color: var(--muted);
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-size: 0.68rem;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
  }}

  th:hover {{ color: var(--accent); }}

  td {{
    padding: 0.65rem 1rem;
    border-top: 1px solid var(--border);
    color: var(--text);
  }}

  .run-name {{
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: var(--muted);
    font-size: 0.7rem;
  }}

  tr.best-row td {{ background: rgba(0,229,255,0.04); }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}

  .section-title {{
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
  }}
</style>
</head>
<body>

<h1>YOLO Run Comparison</h1>
<p class="subtitle">// {len(df)} runs validated — sorted by mAP50 descending</p>

<div class="metric-selector">
  <button class="metric-btn active" onclick="setMetric('mAP50')">mAP50</button>
  <button class="metric-btn" onclick="setMetric('mAP50-95')">mAP50-95</button>
  <button class="metric-btn" onclick="setMetric('Precision')">Precision</button>
  <button class="metric-btn" onclick="setMetric('Recall')">Recall</button>
  <button class="metric-btn" onclick="setMetric('F1')">F1</button>
</div>

<div class="chart-panel">
  <span class="sort-hint">sorted by selected metric ↓</span>
  <canvas id="barChart"></canvas>
</div>

<p class="section-title">// Full results table</p>
<table id="resultsTable">
  <thead>
    <tr>
      <th>#</th>
      <th>Run</th>
      <th onclick="sortTable(2)">mAP50 ↕</th>
      <th onclick="sortTable(3)">mAP50-95 ↕</th>
      <th onclick="sortTable(4)">Precision ↕</th>
      <th onclick="sortTable(5)">Recall ↕</th>
      <th onclick="sortTable(6)">F1 ↕</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<script>
const DATA = {{
  'mAP50':    {{ values: {map50_json},   label: 'mAP50',    color: '#00e5ff' }},
  'mAP50-95': {{ values: {map5095_json}, label: 'mAP50-95', color: '#c084fc' }},
  'Precision':{{ values: {prec_json},    label: 'Precision', color: '#a0ff6f' }},
  'Recall':   {{ values: {rec_json},     label: 'Recall',    color: '#ffb347' }},
  'F1':       {{ values: {f1_json},      label: 'F1',        color: '#ff4d6d' }},
}};

const RUNS = {runs_json};

let currentMetric = 'mAP50';
let chart;

function buildChart(metric) {{
  const d = DATA[metric];
  // Sort by value descending
  const indices = [...d.values.keys()].sort((a,b) => d.values[b] - d.values[a]);
  const labels  = indices.map(i => RUNS[i].replace(/^run_\d+_/, ''));
  const values  = indices.map(i => d.values[i]);

  if (chart) chart.destroy();

  chart = new Chart(document.getElementById('barChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [{{
        label: d.label,
        data: values,
        backgroundColor: d.color + '33',
        borderColor: d.color,
        borderWidth: 1.5,
        borderRadius: 4,
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{
          callbacks: {{
            title: (items) => RUNS[indices[items[0].dataIndex]],
            label: (item) => ` ${{d.label}}: ${{item.raw.toFixed(4)}}`
          }}
        }}
      }},
      scales: {{
        x: {{
          ticks: {{ color: '#64748b', font: {{ family: 'JetBrains Mono', size: 9 }}, maxRotation: 45 }},
          grid: {{ color: '#1f2330' }}
        }},
        y: {{
          ticks: {{ color: '#64748b', font: {{ family: 'JetBrains Mono', size: 10 }} }},
          grid: {{ color: '#1f2330' }},
          min: 0, max: 1
        }}
      }}
    }}
  }});
}}

function setMetric(metric) {{
  currentMetric = metric;
  document.querySelectorAll('.metric-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  buildChart(metric);
}}

// Table sort
let sortDir = {{}};
function sortTable(col) {{
  const tbody = document.querySelector('#resultsTable tbody');
  const rows  = Array.from(tbody.querySelectorAll('tr'));
  sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    const av = parseFloat(a.cells[col].textContent) || 0;
    const bv = parseFloat(b.cells[col].textContent) || 0;
    return sortDir[col] ? bv - av : av - bv;
  }});
  rows.forEach(r => tbody.appendChild(r));
}}

buildChart('mAP50');
</script>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    validate_all()