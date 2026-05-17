from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any


MetricRow = dict[str, Any]


def generate_bidirectional_synthesis_report(
    forward_run: str | Path,
    reverse_run: str | Path,
    output_dir: str | Path = "output/corpus_pairwise_bidirectional_synthesis/report",
) -> Path:
    """Generate CSV/JSON/HTML synthesis artifacts for forward and reverse corpus runs."""
    forward_run = Path(forward_run)
    reverse_run = Path(reverse_run)
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = build_synthesis(forward_run, reverse_run)
    summary = build_summary(rows)
    write_csv(report_dir / "synthesis.csv", rows)
    (report_dir / "synthesis.json").write_text(
        json.dumps({"summary": summary, "pairs": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (report_dir / "report_data.js").write_text(
        "window.BIDIRECTIONAL_REPORT_DATA = "
        + json.dumps({"summary": summary, "pairs": rows}, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    (report_dir / "index.html").write_text(build_html(), encoding="utf-8")
    return report_dir / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a bidirectional survival report from forward and reverse corpus pairwise runs."
    )
    parser.add_argument("--forward-run", required=True, help="Forward run directory: SMDG as A, Txt-18 as B.")
    parser.add_argument("--reverse-run", required=True, help="Reverse run directory: Txt-18 as A, SMDG as B.")
    parser.add_argument(
        "--output-dir",
        default="output/corpus_pairwise_bidirectional_synthesis/report",
        help="Directory for index.html, report_data.js, synthesis.csv, and synthesis.json.",
    )
    args = parser.parse_args()

    print(generate_bidirectional_synthesis_report(args.forward_run, args.reverse_run, args.output_dir))


def build_synthesis(forward_run: Path, reverse_run: Path) -> list[MetricRow]:
    forward = read_rows(forward_run / "document_pair_summary.csv")
    reverse = read_rows(reverse_run / "document_pair_summary.csv")
    reverse_by_pair = {(row["doc_b_relative_path"], row["doc_a_relative_path"]): row for row in reverse}
    joined: list[MetricRow] = []

    for row in forward:
        key = (row["doc_a_relative_path"], row["doc_b_relative_path"])
        reverse_row = reverse_by_pair.get(key)
        if reverse_row is None:
            continue
        joined.append(build_pair_row(row, reverse_row))

    add_rank_survival_scores(joined)
    joined.sort(key=lambda row: (-row["overall_survival"], -row["broad_affinity_survival"], row["smdg_short"], row["txt_short"]))
    for index, row in enumerate(joined, start=1):
        row["overall_rank"] = index
        row["signal_label"] = classify_signal(row)
        row["reading_reason"] = reading_reason(row)
    return joined


def build_pair_row(forward: dict[str, str], reverse: dict[str, str]) -> MetricRow:
    f_max = f(forward["max_score"])
    r_max = f(reverse["max_score"])
    f_p95 = f(forward["p95_score"])
    r_p95 = f(reverse["p95_score"])
    f_smdg_to_txt = f(forward["mean_best_a_to_b"])
    r_smdg_to_txt = f(reverse["mean_best_b_to_a"])
    f_txt_to_smdg = f(forward["mean_best_b_to_a"])
    r_txt_to_smdg = f(reverse["mean_best_a_to_b"])

    return {
        "smdg_path": forward["doc_a_relative_path"],
        "txt_path": forward["doc_b_relative_path"],
        "smdg_short": short_name(forward["doc_a_relative_path"]),
        "txt_short": short_name(forward["doc_b_relative_path"]),
        "smdg_label": readable_label(forward["doc_a_relative_path"], "SMDG"),
        "txt_label": readable_label(forward["doc_b_relative_path"], "Txt-18"),
        "forward_pair_id": forward["pair_id"],
        "reverse_pair_id": reverse["pair_id"],
        "smdg_sentence_count": int(forward["sentence_count_a"]),
        "txt_sentence_count": int(forward["sentence_count_b"]),
        "matrix_score_count": int(forward["matrix_score_count"]),
        "f_max": round6(f_max),
        "r_max": round6(r_max),
        "max_min": round6(min(f_max, r_max)),
        "max_delta_abs": round6(abs(r_max - f_max)),
        "f_p95": round6(f_p95),
        "r_p95": round6(r_p95),
        "p95_min": round6(min(f_p95, r_p95)),
        "p95_delta_abs": round6(abs(r_p95 - f_p95)),
        "f_smdg_to_txt": round6(f_smdg_to_txt),
        "r_smdg_to_txt": round6(r_smdg_to_txt),
        "smdg_to_txt_min": round6(min(f_smdg_to_txt, r_smdg_to_txt)),
        "smdg_to_txt_delta_abs": round6(abs(r_smdg_to_txt - f_smdg_to_txt)),
        "f_txt_to_smdg": round6(f_txt_to_smdg),
        "r_txt_to_smdg": round6(r_txt_to_smdg),
        "txt_to_smdg_min": round6(min(f_txt_to_smdg, r_txt_to_smdg)),
        "txt_to_smdg_delta_abs": round6(abs(r_txt_to_smdg - f_txt_to_smdg)),
    }


def add_rank_survival_scores(rows: list[MetricRow]) -> None:
    score_specs = {
        "local_spike_survival": ("f_max", "r_max"),
        "broad_affinity_survival": ("f_p95", "r_p95"),
        "smdg_to_txt_survival": ("f_smdg_to_txt", "r_smdg_to_txt"),
        "txt_to_smdg_survival": ("f_txt_to_smdg", "r_txt_to_smdg"),
    }
    percentile_by_metric = {metric: percentile_scores(rows, metric) for metrics in score_specs.values() for metric in metrics}

    for row in rows:
        for output_metric, (forward_metric, reverse_metric) in score_specs.items():
            row[output_metric] = round6(
                min(
                    percentile_by_metric[forward_metric][row_key(row)],
                    percentile_by_metric[reverse_metric][row_key(row)],
                )
            )
        row["direction_balance"] = round6(1.0 - abs(row["smdg_to_txt_survival"] - row["txt_to_smdg_survival"]))
        row["overall_survival"] = round6(
            0.35 * row["broad_affinity_survival"]
            + 0.25 * row["smdg_to_txt_survival"]
            + 0.25 * row["txt_to_smdg_survival"]
            + 0.15 * row["local_spike_survival"]
        )


def percentile_scores(rows: list[MetricRow], metric: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (-float(row[metric]), row_key(row)))
    count = len(ordered)
    return {row_key(row): (count - rank + 1) / count for rank, row in enumerate(ordered, start=1)}


def build_summary(rows: list[MetricRow]) -> dict[str, Any]:
    top10 = rows[:10]
    return {
        "pair_count": len(rows),
        "smdg_count": len({row["smdg_path"] for row in rows}),
        "txt_count": len({row["txt_path"] for row in rows}),
        "top_overall_mean": round6(statistics.mean(row["overall_survival"] for row in top10)) if top10 else None,
        "top_broad_pair": top_pair(rows, "broad_affinity_survival"),
        "top_local_pair": top_pair(rows, "local_spike_survival"),
        "top_smdg_to_txt_pair": top_pair(rows, "smdg_to_txt_survival"),
        "top_txt_to_smdg_pair": top_pair(rows, "txt_to_smdg_survival"),
        "mean_forward_reverse_max_delta": mean_metric(rows, "max_delta_abs"),
        "mean_forward_reverse_p95_delta": mean_metric(rows, "p95_delta_abs"),
        "mean_smdg_to_txt_delta": mean_metric(rows, "smdg_to_txt_delta_abs"),
        "mean_txt_to_smdg_delta": mean_metric(rows, "txt_to_smdg_delta_abs"),
    }


def top_pair(rows: list[MetricRow], metric: str) -> dict[str, Any] | None:
    if not rows:
        return None
    row = max(rows, key=lambda item: (item[metric], item["overall_survival"]))
    return {
        "smdg_short": row["smdg_short"],
        "txt_short": row["txt_short"],
        "score": row[metric],
    }


def classify_signal(row: MetricRow) -> str:
    scores = {
        "Broad bidirectional affinity": row["broad_affinity_survival"],
        "SMDG covers Txt-18": row["smdg_to_txt_survival"],
        "Txt-18 covered by SMDG": row["txt_to_smdg_survival"],
        "Local sentence spike": row["local_spike_survival"],
    }
    top_label, top_score = max(scores.items(), key=lambda item: item[1])
    low = min(scores.values())
    if row["overall_survival"] >= 0.9 and low >= 0.75:
        return "Robust all-around"
    if top_score >= 0.85:
        return top_label
    return "Mixed signal"


def reading_reason(row: MetricRow) -> str:
    if row["signal_label"] == "Robust all-around":
        return "High broad, local, and directional survival; this is a strong first-pass close-reading candidate."
    if row["signal_label"] == "Broad bidirectional affinity":
        return "The upper tail of the full matrix survives both model directions, suggesting repeated related passages rather than one isolated match."
    if row["signal_label"] == "Local sentence spike":
        return "A very high best sentence pair survives both directions; inspect for quotation, formula, or shared stock phrase."
    if row["signal_label"] == "SMDG covers Txt-18":
        return "Many SMDG sentences find strong Txt-18 matches in both runs; useful for testing whether SMDG draws on that text family."
    if row["signal_label"] == "Txt-18 covered by SMDG":
        return "Many Txt-18 sentences find strong SMDG matches in both runs; useful for testing whether a shorter source is represented inside SMDG."
    return "Worth checking only after stronger survival candidates, or as a contrast case."


def mean_metric(rows: list[MetricRow], metric: str) -> float | None:
    if not rows:
        return None
    return round6(statistics.mean(float(row[metric]) for row in rows))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[MetricRow]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def row_key(row: MetricRow) -> str:
    return f"{row['smdg_path']}||{row['txt_path']}"


def short_name(path: str) -> str:
    return Path(path).name.removesuffix(".txt")


def readable_label(path: str, corpus: str) -> str:
    name = short_name(path)
    if corpus == "SMDG":
        match = re.match(r"(?P<num>\d+[a-z]?)-SMDG-(?P<title>.+)", name)
        if match:
            return f"SMDG {match.group('num')} | {humanize_title(match.group('title'))}"
        return f"SMDG | {humanize_title(name)}"

    version_note = "older witness | " if path.startswith("0-Older-versions/") else ""
    siglum_match = re.match(r"(?P<siglum>LL?\d+)[-_]?(?P<title>.*)", name)
    if siglum_match:
        title = siglum_match.group("title") or name
        return f"Txt-18 {version_note}{siglum_match.group('siglum')} | {humanize_title(title)}"
    return f"Txt-18 {version_note}| {humanize_title(name)}"


def humanize_title(value: str) -> str:
    value = value.replace("_", " ").replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.replace("་", "")
    return value.strip()


def f(value: str) -> float:
    return float(value)


def round6(value: float) -> float:
    return round(float(value), 6)


def build_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bidirectional Pair Survival Report</title>
  <script src="report_data.js"></script>
  <style>
    :root {
      color-scheme: light;
      --ink: #171717;
      --muted: #60646c;
      --line: #d8dce2;
      --paper: #f7f8fa;
      --panel: #ffffff;
      --accent: #0f766e;
      --accent-2: #7c2d12;
      --heat: #b45309;
      --cool: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
      line-height: 1.45;
    }
    header, main, footer { max-width: 1280px; margin: 0 auto; padding: 24px; }
    header { padding-top: 36px; }
    .top-link { display: inline-block; margin-bottom: 10px; color: var(--accent); font-size: 13px; font-weight: 700; text-decoration: none; }
    h1 { margin: 0 0 10px; font-size: clamp(30px, 4vw, 52px); letter-spacing: 0; }
    h2 { margin: 30px 0 12px; font-size: 22px; }
    p { margin: 0 0 12px; color: var(--muted); max-width: 900px; }
    .summary {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px;
      margin: 22px 0;
    }
    .metric, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }
    .metric strong { display: block; font-size: 26px; }
    .metric span { color: var(--muted); font-size: 13px; }
    .controls {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 220px 220px;
      gap: 12px;
      align-items: end;
      margin: 18px 0;
    }
    label { display: grid; gap: 6px; font-size: 13px; color: var(--muted); }
    input, select {
      width: 100%;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid var(--line); }
    th, td { border-bottom: 1px solid var(--line); padding: 10px; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #eef2f4; z-index: 1; font-size: 12px; color: #343840; }
    tbody tr:hover { background: #f9fafb; }
    .pair-title { font-weight: 700; font-size: 15px; }
    .corpus-label {
      display: inline-flex;
      min-width: 54px;
      justify-content: center;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 1px 5px;
      margin-right: 6px;
      color: #334155;
      background: #f8fafc;
      font-size: 11px;
      font-weight: 700;
    }
    .pair-doc { margin-bottom: 8px; }
    .file-provenance {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .score { font-variant-numeric: tabular-nums; font-weight: 700; }
    .pill {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      background: #f8fafc;
      color: #334155;
      white-space: nowrap;
    }
    .bars { display: grid; gap: 6px; min-width: 180px; }
    .bar { display: grid; grid-template-columns: 68px 1fr 44px; gap: 8px; align-items: center; font-size: 12px; }
    .track { height: 8px; background: #e5e7eb; border-radius: 99px; overflow: hidden; }
    .fill { height: 100%; background: var(--accent); }
    .fill.local { background: var(--heat); }
    .fill.dir { background: var(--cool); }
    .note { color: var(--muted); max-width: 320px; }
    details { max-width: 520px; }
    summary { cursor: pointer; color: var(--accent); font-weight: 650; }
    .details-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }
    .explain {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
    }
    .explain h3 { margin: 0 0 6px; font-size: 16px; }
    @media (max-width: 860px) {
      header, main, footer { padding: 18px; }
      .controls { grid-template-columns: 1fr; }
      table { display: block; overflow-x: auto; }
      th { position: static; }
    }
  </style>
</head>
<body>
  <header>
    <a class="top-link" href="../">Reports index</a>
    <h1>Bidirectional Pair Survival Report</h1>
    <p>This report joins the forward run (SMDG as query side, Txt-18 as corpus side) with the reverse run (Txt-18 as query side, SMDG as corpus side). It prioritizes file pairs whose signal remains high in both model prompt directions.</p>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <section class="panel">
      <h2>How To Read This</h2>
      <div class="explain">
        <div>
          <h3>Overall survival</h3>
          <p>A rank-based composite. A pair scores high only if it remains strong when compared in both model directions. This is the close-reading queue.</p>
        </div>
        <div>
          <h3>Broad affinity</h3>
          <p>Uses the p95 score in both runs. High values suggest many strong cells in the matrix, not just one spectacular sentence match.</p>
        </div>
        <div>
          <h3>Local spike</h3>
          <p>Uses the single best sentence-pair score in both runs. High values point to possible quotation, formulaic phrasing, or a concentrated parallel.</p>
        </div>
        <div>
          <h3>Directional coverage</h3>
          <p>SMDG→Txt asks how well SMDG sentences find matches in Txt-18. Txt→SMDG asks the inverse. Asymmetry is expected because each side has different length and internal repetition.</p>
        </div>
      </div>
    </section>

    <section>
      <h2>Prioritized Pairs</h2>
      <div class="controls">
        <label>Search file names
          <input id="search" placeholder="Try: rig-pa, LL03, thig-le">
        </label>
        <label>Sort
          <select id="sort">
            <option value="overall_survival">Overall survival</option>
            <option value="broad_affinity_survival">Broad affinity</option>
            <option value="local_spike_survival">Local spike</option>
            <option value="smdg_to_txt_survival">SMDG → Txt coverage</option>
            <option value="txt_to_smdg_survival">Txt → SMDG coverage</option>
            <option value="p95_min">p95 raw floor</option>
            <option value="max_min">max raw floor</option>
          </select>
        </label>
        <label>Rows
          <select id="limit">
            <option value="25">Top 25</option>
            <option value="50">Top 50</option>
            <option value="100">Top 100</option>
            <option value="352">All pairs</option>
          </select>
        </label>
      </div>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Pair</th>
            <th>Signal</th>
            <th>Survival Scores</th>
            <th>Raw Floors</th>
            <th>Why Read</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </main>
  <footer>
    <p>Generated from summary artifacts only: forward and reverse document-pair CSVs. The sentence matrices and top-k evidence remain in the original run directories.</p>
  </footer>
<script>
const data = window.BIDIRECTIONAL_REPORT_DATA;
const pairs = data.pairs;
const fmt = (value, digits = 3) => Number(value).toFixed(digits);

function renderSummary() {
  const items = [
    ['Pairs joined', data.summary.pair_count],
    ['SMDG files', data.summary.smdg_count],
    ['Txt-18 files', data.summary.txt_count],
    ['Mean max delta', fmt(data.summary.mean_forward_reverse_max_delta)],
    ['Mean p95 delta', fmt(data.summary.mean_forward_reverse_p95_delta)],
  ];
  document.getElementById('summary').innerHTML = items.map(([label, value]) =>
    `<div class="metric"><strong>${value}</strong><span>${label}</span></div>`
  ).join('');
}

function bar(label, value, cls = '') {
  return `<div class="bar"><span>${label}</span><div class="track"><div class="fill ${cls}" style="width:${Math.max(0, Math.min(1, value)) * 100}%"></div></div><span>${fmt(value)}</span></div>`;
}

function metricDetails(row) {
  return `<details>
    <summary>metric details</summary>
    <div class="details-grid">
      <div>f max ${fmt(row.f_max)} / r max ${fmt(row.r_max)}</div>
      <div>max delta ${fmt(row.max_delta_abs)}</div>
      <div>f p95 ${fmt(row.f_p95)} / r p95 ${fmt(row.r_p95)}</div>
      <div>p95 delta ${fmt(row.p95_delta_abs)}</div>
      <div>f SMDG→Txt ${fmt(row.f_smdg_to_txt)} / r SMDG→Txt ${fmt(row.r_smdg_to_txt)}</div>
      <div>SMDG→Txt delta ${fmt(row.smdg_to_txt_delta_abs)}</div>
      <div>f Txt→SMDG ${fmt(row.f_txt_to_smdg)} / r Txt→SMDG ${fmt(row.r_txt_to_smdg)}</div>
      <div>Txt→SMDG delta ${fmt(row.txt_to_smdg_delta_abs)}</div>
    </div>
  </details>`;
}

function fileDetails(row) {
  return `<details class="file-provenance">
    <summary>exact filenames</summary>
    <div><strong>SMDG:</strong> ${row.smdg_path}</div>
    <div><strong>Txt-18:</strong> ${row.txt_path}</div>
    <div><strong>Run pair IDs:</strong> forward ${row.forward_pair_id}, reverse ${row.reverse_pair_id}</div>
  </details>`;
}

function renderRows() {
  const query = document.getElementById('search').value.trim().toLowerCase();
  const sortKey = document.getElementById('sort').value;
  const limit = Number(document.getElementById('limit').value);
  const filtered = pairs
    .filter(row => !query || `${row.smdg_label} ${row.txt_label} ${row.smdg_short} ${row.txt_short} ${row.smdg_path} ${row.txt_path}`.toLowerCase().includes(query))
    .sort((a, b) => b[sortKey] - a[sortKey] || b.overall_survival - a.overall_survival)
    .slice(0, limit);

  document.getElementById('rows').innerHTML = filtered.map((row, idx) => `
    <tr>
      <td class="score">${idx + 1}</td>
      <td>
        <div class="pair-doc">
          <div class="pair-title"><span class="corpus-label">SMDG</span>${row.smdg_label}</div>
          <div class="sub">${row.smdg_short}</div>
        </div>
        <div class="pair-doc">
          <div class="pair-title"><span class="corpus-label">Txt-18</span>${row.txt_label}</div>
          <div class="sub">${row.txt_short}</div>
        </div>
        <div class="sub">${row.smdg_sentence_count} × ${row.txt_sentence_count} sentences</div>
        ${fileDetails(row)}
      </td>
      <td><span class="pill">${row.signal_label}</span><div class="sub">overall rank ${row.overall_rank}</div></td>
      <td class="bars">
        ${bar('overall', row.overall_survival)}
        ${bar('broad', row.broad_affinity_survival)}
        ${bar('local', row.local_spike_survival, 'local')}
        ${bar('S→T', row.smdg_to_txt_survival, 'dir')}
        ${bar('T→S', row.txt_to_smdg_survival, 'dir')}
      </td>
      <td>
        <div class="score">p95 ${fmt(row.p95_min)}</div>
        <div class="score">max ${fmt(row.max_min)}</div>
        <div class="sub">direction floors: ${fmt(row.smdg_to_txt_min)} / ${fmt(row.txt_to_smdg_min)}</div>
        ${metricDetails(row)}
      </td>
      <td class="note">${row.reading_reason}</td>
    </tr>
  `).join('');
}

renderSummary();
renderRows();
document.getElementById('search').addEventListener('input', renderRows);
document.getElementById('sort').addEventListener('change', renderRows);
document.getElementById('limit').addEventListener('change', renderRows);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
