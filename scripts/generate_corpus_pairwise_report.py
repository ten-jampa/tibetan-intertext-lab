from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a local interactive HTML report for a corpus pairwise run."
    )
    parser.add_argument("--run-dir", required=True, help="Corpus pairwise output directory.")
    parser.add_argument("--output-dir", help="Report directory. Defaults to <run-dir>/report.")
    parser.add_argument("--heatmap-size", type=int, default=56, help="Downsampled pair heatmap size.")
    parser.add_argument("--max-topk", type=int, default=100, help="Top-k rows to embed per pair.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_report_data(run_dir, args.heatmap_size, args.max_topk)
    (output_dir / "report_data.js").write_text(
        "window.CORPUS_REPORT_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(build_html(), encoding="utf-8")
    print(output_dir / "index.html")


def build_report_data(run_dir: Path, heatmap_size: int, max_topk: int) -> dict[str, Any]:
    manifest = json.loads((run_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    docs_a = read_csv(run_dir / "documents_a.csv")
    docs_b = read_csv(run_dir / "documents_b.csv")
    pairs = read_csv(run_dir / "document_pair_summary.csv")

    for rows in (docs_a, docs_b):
        for row in rows:
            row["sentence_count"] = int(row["sentence_count"])
            row["short_name"] = shorten_path(row["relative_path"])
            row["embedding_shape"] = list(np.load(resolve_run_path(run_dir, row["embeddings_npy"]), mmap_mode="r").shape)

    pair_payloads: list[dict[str, Any]] = []
    for row in pairs:
        pair_dir = run_dir / "pairs" / row["pair_id"]
        matrix = np.load(pair_dir / "similarity_matrix.npy", mmap_mode="r")
        topk_rows = read_csv(pair_dir / "topk_pairs.csv")[:max_topk]
        pair_payloads.append(
            {
                "pair_id": row["pair_id"],
                "doc_a_id": row["doc_a_id"],
                "doc_b_id": row["doc_b_id"],
                "doc_a_name": shorten_path(row["doc_a_relative_path"]),
                "doc_b_name": shorten_path(row["doc_b_relative_path"]),
                "doc_a_relative_path": row["doc_a_relative_path"],
                "doc_b_relative_path": row["doc_b_relative_path"],
                "sentence_count_a": int(row["sentence_count_a"]),
                "sentence_count_b": int(row["sentence_count_b"]),
                "score_count": int(row["matrix_score_count"]),
                "max_score": round_float(row["max_score"]),
                "mean_score": round_float(row["mean_score"]),
                "median_score": round_float(row["median_score"]),
                "p95_score": round_float(row["p95_score"]),
                "mean_best_a_to_b": round_float(row["mean_best_a_to_b"]),
                "mean_best_b_to_a": round_float(row["mean_best_b_to_a"]),
                "top_k_returned": int(row["top_k_returned"]),
                "similarity_npy": row["similarity_npy"],
                "topk_csv": row["topk_csv"],
                "topk": normalize_topk(topk_rows),
                "heatmap": downsample_matrix(matrix, heatmap_size),
            }
        )

    gpu = read_gpu_monitor(run_dir / "logs" / "gpu_monitor.csv")
    stderr = (run_dir / "logs" / "full_run.stderr.log").read_text(encoding="utf-8", errors="replace")
    timing = parse_time_log(stderr)

    return {
        "run": {
            "title": "Full Corpus Pairwise Run",
            "model_id": manifest.get("model_id"),
            "device": manifest.get("device"),
            "batch_size": manifest.get("batch_size"),
            "pair_count": manifest.get("pair_count"),
            "doc_count_a": len(docs_a),
            "doc_count_b": len(docs_b),
            "total_sentences": sum(row["sentence_count"] for row in docs_a + docs_b),
            "embedding_dim": docs_a[0]["embedding_shape"][1] if docs_a else None,
            "wall_time": timing.get("wall_time"),
            "max_resident_set_kb": timing.get("max_resident_set_kb"),
            "max_gpu_mem_mib": max((row["mem_used_mib"] for row in gpu), default=None),
            "max_gpu_util_pct": max((row["gpu_util_pct"] for row in gpu), default=None),
            "heatmap_size": heatmap_size,
        },
        "documentsA": docs_a,
        "documentsB": docs_b,
        "pairs": pair_payloads,
        "gpu": gpu,
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_run_path(run_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path
    try:
        return run_dir / path.relative_to(run_dir.as_posix())
    except ValueError:
        return run_dir / path.name


def shorten_path(path: str) -> str:
    name = Path(path).name
    return re.sub(r"\.txt$", "", name)


def round_float(value: str | float) -> float:
    return round(float(value), 6)


def normalize_topk(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "rank": int(row["rank"]),
                "score": round_float(row["score"]),
                "i": int(row["i"]),
                "j": int(row["j"]),
                "sentence_a": row["sentence_a"],
                "sentence_b": row["sentence_b"],
            }
        )
    return normalized


def downsample_matrix(matrix: np.ndarray, size: int) -> list[list[float]]:
    rows, cols = matrix.shape
    row_edges = np.linspace(0, rows, min(size, rows) + 1, dtype=int)
    col_edges = np.linspace(0, cols, min(size, cols) + 1, dtype=int)
    sampled: list[list[float]] = []
    for r0, r1 in zip(row_edges[:-1], row_edges[1:]):
        out_row = []
        for c0, c1 in zip(col_edges[:-1], col_edges[1:]):
            block = matrix[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
            out_row.append(round(float(np.mean(block)), 4))
        sampled.append(out_row)
    return sampled


def read_gpu_monitor(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        try:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "gpu_util_pct": int(row["gpu_util_pct"].strip()),
                    "mem_used_mib": int(row["mem_used_mib"].strip()),
                    "mem_total_mib": int(row["mem_total_mib"].strip()),
                }
            )
        except (KeyError, ValueError):
            continue
    return rows


def parse_time_log(text: str) -> dict[str, Any]:
    wall = re.search(r"Elapsed \\(wall clock\\) time.*: ([^\\n]+)", text)
    rss = re.search(r"Maximum resident set size \\(kbytes\\): ([0-9]+)", text)
    return {
        "wall_time": wall.group(1).strip() if wall else None,
        "max_resident_set_kb": int(rss.group(1)) if rss else None,
    }


def build_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tibetan Corpus Pairwise Report</title>
  <script src="report_data.js"></script>
  <style>
    :root {
      --ink: #171717;
      --muted: #666;
      --line: #d8d8d8;
      --panel: #f7f7f5;
      --accent: #245c63;
      --accent-2: #a6422a;
      --gold: #b3852f;
      --bg: #fffffc;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    header { padding: 26px 32px 18px; border-bottom: 1px solid var(--line); background: #fff; }
    h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 18px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 15px; letter-spacing: 0; }
    .sub { color: var(--muted); font-size: 14px; }
    main { display: grid; grid-template-columns: minmax(320px, 420px) 1fr; min-height: calc(100vh - 93px); }
    aside { border-right: 1px solid var(--line); padding: 18px; background: var(--panel); overflow: auto; max-height: calc(100vh - 93px); }
    section { padding: 22px 26px 36px; overflow: auto; max-height: calc(100vh - 93px); }
    .stats { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-bottom: 18px; }
    .stat { border: 1px solid var(--line); background: #fff; padding: 10px; border-radius: 6px; min-height: 64px; }
    .stat b { display: block; font-size: 20px; margin-bottom: 2px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .controls { display: grid; gap: 10px; margin-bottom: 14px; }
    label { display: grid; gap: 4px; font-size: 12px; color: var(--muted); }
    input, select { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); padding: 8px 10px; font-size: 14px; }
    button { border: 1px solid var(--line); border-radius: 6px; background: #fff; color: var(--ink); padding: 8px 10px; font-size: 13px; cursor: pointer; }
    button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .metric-buttons { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; }
    .pair-list { display: grid; gap: 6px; }
    .pair-row { border: 1px solid var(--line); background: #fff; border-radius: 6px; padding: 9px; cursor: pointer; }
    .pair-row.active { border-color: var(--accent); box-shadow: inset 4px 0 0 var(--accent); }
    .pair-title { font-weight: 700; font-size: 13px; margin-bottom: 4px; }
    .pair-meta { color: var(--muted); font-size: 12px; display: flex; gap: 8px; flex-wrap: wrap; }
    .grid { display: grid; gap: 18px; }
    .panel { border: 1px solid var(--line); border-radius: 8px; background: #fff; padding: 16px; }
    .two { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }
    .canvas-wrap { width: 100%; overflow: auto; }
    canvas { display: block; max-width: 100%; border: 1px solid var(--line); background: #fff; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; background: #fafafa; position: sticky; top: 0; }
    .matches { display: grid; gap: 10px; }
    .match { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fff; }
    .match-head { display: flex; justify-content: space-between; gap: 12px; color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .sentences { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .sentence { font-size: 15px; line-height: 1.6; padding: 10px; border-radius: 6px; background: var(--panel); }
    .note { color: var(--muted); font-size: 12px; line-height: 1.45; }
    .bar { height: 8px; background: #e9e4d9; border-radius: 999px; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: var(--accent); }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; }
      aside, section { max-height: none; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .two, .sentences { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Tibetan Corpus Pairwise Report</h1>
    <div class="sub" id="runSubtitle"></div>
  </header>
  <main>
    <aside>
      <div class="stats" id="stats"></div>
      <div class="controls">
        <label>Search pair or file
          <input id="search" placeholder="A001, rig-pa, SMDG...">
        </label>
        <label>Sort pairs by
          <select id="sortMetric">
            <option value="max_score">Max score</option>
            <option value="mean_best_a_to_b">Mean best A to B</option>
            <option value="mean_best_b_to_a">Mean best B to A</option>
            <option value="p95_score">P95 score</option>
            <option value="mean_score">Mean score</option>
            <option value="score_count">Matrix size</option>
          </select>
        </label>
        <label>Top K shown for selected pair
          <input id="topK" type="number" min="1" max="100" value="20">
        </label>
        <div class="metric-buttons">
          <button class="active" data-overview="max_score">Overview: max</button>
          <button data-overview="mean_best_a_to_b">Overview: best A→B</button>
          <button data-overview="mean_best_b_to_a">Overview: best B→A</button>
          <button data-overview="p95_score">Overview: p95</button>
        </div>
      </div>
      <h2>Corpus Heatmap</h2>
      <div class="canvas-wrap"><canvas id="overviewCanvas" width="360" height="260"></canvas></div>
      <p class="note">Rows are SMDG docs, columns are Txt-18 docs. Click a cell or a pair row to inspect details.</p>
      <h2>Pairs</h2>
      <div class="pair-list" id="pairList"></div>
    </aside>
    <section>
      <div class="grid">
        <div class="panel" id="pairHeader"></div>
        <div class="two">
          <div class="panel">
            <h2>Selected Pair Heatmap</h2>
            <div class="canvas-wrap"><canvas id="pairCanvas" width="620" height="420"></canvas></div>
            <p class="note">This is a downsampled view of the full sentence-by-sentence matrix. Full `.npy` paths remain in the pair artifacts.</p>
          </div>
          <div class="panel">
            <h2>Run Resources</h2>
            <div id="resources"></div>
            <canvas id="gpuCanvas" width="620" height="180"></canvas>
          </div>
        </div>
        <div class="panel">
          <h2>Top Matches</h2>
          <div class="matches" id="matches"></div>
        </div>
        <div class="panel">
          <h2>Documents</h2>
          <div class="two">
            <div><h3>SMDG</h3><div id="docsA"></div></div>
            <div><h3>Txt-18</h3><div id="docsB"></div></div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const DATA = window.CORPUS_REPORT_DATA;
    let selectedPair = DATA.pairs[0];
    let overviewMetric = 'max_score';

    const fmt = new Intl.NumberFormat('en-US');
    const score = (x) => Number(x).toFixed(3);
    const $ = (id) => document.getElementById(id);

    function colorFor(v, min, max) {
      const t = max === min ? 0 : Math.max(0, Math.min(1, (v - min) / (max - min)));
      const stops = [
        [248, 248, 240],
        [223, 184, 91],
        [164, 66, 42],
        [36, 92, 99]
      ];
      const p = t * (stops.length - 1);
      const i = Math.min(stops.length - 2, Math.floor(p));
      const f = p - i;
      const c = stops[i].map((x, k) => Math.round(x + (stops[i + 1][k] - x) * f));
      return `rgb(${c[0]},${c[1]},${c[2]})`;
    }

    function init() {
      $('runSubtitle').textContent = `${DATA.run.model_id} · ${DATA.run.device} · ${DATA.run.doc_count_a} x ${DATA.run.doc_count_b} docs · ${fmt.format(DATA.run.total_sentences)} sentence embeddings`;
      renderStats();
      renderResources();
      renderDocs();
      bindControls();
      renderPairList();
      renderOverview();
      selectPair(DATA.pairs[0].pair_id);
    }

    function renderStats() {
      const rssGb = DATA.run.max_resident_set_kb ? (DATA.run.max_resident_set_kb / 1024 / 1024).toFixed(1) + ' GB' : 'n/a';
      const stats = [
        [DATA.run.pair_count, 'document pairs'],
        [DATA.run.total_sentences, 'sentence embeddings'],
        [DATA.run.embedding_dim, 'embedding dimensions'],
        [DATA.run.wall_time || 'n/a', 'wall time'],
        [DATA.run.max_gpu_mem_mib + ' MiB', 'peak GPU memory'],
        [rssGb, 'peak host RSS']
      ];
      $('stats').innerHTML = stats.map(([v, label]) => `<div class="stat"><b>${formatValue(v)}</b><span>${label}</span></div>`).join('');
    }

    function renderResources() {
      $('resources').innerHTML = `
        <table>
          <tr><th>Model</th><td>${escapeHtml(DATA.run.model_id)}</td></tr>
          <tr><th>Device</th><td>${DATA.run.device}, batch ${DATA.run.batch_size}</td></tr>
          <tr><th>Peak GPU util</th><td>${DATA.run.max_gpu_util_pct}%</td></tr>
          <tr><th>Peak GPU memory</th><td>${DATA.run.max_gpu_mem_mib} MiB</td></tr>
        </table>`;
      drawGpu();
    }

    function bindControls() {
      $('search').addEventListener('input', renderPairList);
      $('sortMetric').addEventListener('change', renderPairList);
      $('topK').addEventListener('input', renderMatches);
      document.querySelectorAll('[data-overview]').forEach(btn => {
        btn.addEventListener('click', () => {
          document.querySelectorAll('[data-overview]').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          overviewMetric = btn.dataset.overview;
          renderOverview();
        });
      });
    }

    function filteredPairs() {
      const q = $('search').value.trim().toLowerCase();
      const metric = $('sortMetric').value;
      return DATA.pairs
        .filter(p => !q || [p.pair_id, p.doc_a_name, p.doc_b_name, p.doc_a_relative_path, p.doc_b_relative_path].join(' ').toLowerCase().includes(q))
        .sort((a, b) => Number(b[metric]) - Number(a[metric]));
    }

    function renderPairList() {
      const rows = filteredPairs().slice(0, 80);
      $('pairList').innerHTML = rows.map(p => `
        <div class="pair-row ${p.pair_id === selectedPair?.pair_id ? 'active' : ''}" data-pair="${p.pair_id}">
          <div class="pair-title">${p.pair_id}: ${escapeHtml(p.doc_a_name)} ↔ ${escapeHtml(p.doc_b_name)}</div>
          <div class="pair-meta"><span>max ${score(p.max_score)}</span><span>p95 ${score(p.p95_score)}</span><span>${fmt.format(p.score_count)} cells</span></div>
        </div>`).join('');
      document.querySelectorAll('[data-pair]').forEach(el => el.addEventListener('click', () => selectPair(el.dataset.pair)));
    }

    function selectPair(pairId) {
      selectedPair = DATA.pairs.find(p => p.pair_id === pairId) || DATA.pairs[0];
      renderPairList();
      renderPairHeader();
      renderPairHeatmap();
      renderMatches();
    }

    function renderPairHeader() {
      const p = selectedPair;
      $('pairHeader').innerHTML = `
        <h2>${p.pair_id}</h2>
        <div class="two">
          <div><h3>${p.doc_a_id}</h3><div>${escapeHtml(p.doc_a_relative_path)}</div><p class="note">${fmt.format(p.sentence_count_a)} sentences</p></div>
          <div><h3>${p.doc_b_id}</h3><div>${escapeHtml(p.doc_b_relative_path)}</div><p class="note">${fmt.format(p.sentence_count_b)} sentences</p></div>
        </div>
        <table>
          <tr><th>Max</th><th>P95</th><th>Mean</th><th>Mean best A→B</th><th>Mean best B→A</th><th>Matrix cells</th></tr>
          <tr><td>${score(p.max_score)}</td><td>${score(p.p95_score)}</td><td>${score(p.mean_score)}</td><td>${score(p.mean_best_a_to_b)}</td><td>${score(p.mean_best_b_to_a)}</td><td>${fmt.format(p.score_count)}</td></tr>
        </table>`;
    }

    function renderOverview() {
      const canvas = $('overviewCanvas');
      const ctx = canvas.getContext('2d');
      const margin = { left: 48, right: 12, top: 18, bottom: 42 };
      const w = canvas.width - margin.left - margin.right;
      const h = canvas.height - margin.top - margin.bottom;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const values = DATA.pairs.map(p => Number(p[overviewMetric]));
      const min = Math.min(...values), max = Math.max(...values);
      const cw = w / DATA.documentsB.length;
      const ch = h / DATA.documentsA.length;
      DATA.pairs.forEach(p => {
        const ai = DATA.documentsA.findIndex(d => d.doc_id === p.doc_a_id);
        const bi = DATA.documentsB.findIndex(d => d.doc_id === p.doc_b_id);
        ctx.fillStyle = colorFor(Number(p[overviewMetric]), min, max);
        ctx.fillRect(margin.left + bi * cw, margin.top + ai * ch, Math.ceil(cw), Math.ceil(ch));
        if (p.pair_id === selectedPair?.pair_id) {
          ctx.strokeStyle = '#111';
          ctx.lineWidth = 2;
          ctx.strokeRect(margin.left + bi * cw + 1, margin.top + ai * ch + 1, cw - 2, ch - 2);
        }
      });
      ctx.fillStyle = '#333';
      ctx.font = '11px system-ui';
      DATA.documentsA.forEach((d, i) => ctx.fillText(d.doc_id, 8, margin.top + i * ch + ch * .68));
      DATA.documentsB.forEach((d, i) => {
        ctx.save();
        ctx.translate(margin.left + i * cw + cw * .55, canvas.height - 8);
        ctx.rotate(-Math.PI / 3);
        ctx.fillText(d.doc_id, 0, 0);
        ctx.restore();
      });
      canvas.onclick = (event) => {
        const rect = canvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) * (canvas.width / rect.width) - margin.left;
        const y = (event.clientY - rect.top) * (canvas.height / rect.height) - margin.top;
        const bi = Math.floor(x / cw), ai = Math.floor(y / ch);
        if (ai >= 0 && ai < DATA.documentsA.length && bi >= 0 && bi < DATA.documentsB.length) {
          selectPair(`${DATA.documentsA[ai].doc_id}__${DATA.documentsB[bi].doc_id}`);
        }
      };
    }

    function renderPairHeatmap() {
      const canvas = $('pairCanvas');
      const ctx = canvas.getContext('2d');
      const hm = selectedPair.heatmap;
      const rows = hm.length, cols = hm[0].length;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const flat = hm.flat();
      const min = Math.min(...flat), max = Math.max(...flat);
      const margin = { left: 58, right: 18, top: 18, bottom: 42 };
      const w = canvas.width - margin.left - margin.right;
      const h = canvas.height - margin.top - margin.bottom;
      const cw = w / cols, ch = h / rows;
      hm.forEach((row, r) => row.forEach((v, c) => {
        ctx.fillStyle = colorFor(v, min, max);
        ctx.fillRect(margin.left + c * cw, margin.top + r * ch, Math.ceil(cw), Math.ceil(ch));
      }));
      ctx.fillStyle = '#333';
      ctx.font = '12px system-ui';
      ctx.fillText(`A sentences (${selectedPair.sentence_count_a})`, margin.left, canvas.height - 12);
      ctx.save();
      ctx.translate(16, margin.top + h);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(`B sentences (${selectedPair.sentence_count_b})`, 0, 0);
      ctx.restore();
      ctx.fillText(`mean ${score(selectedPair.mean_score)} · max ${score(selectedPair.max_score)}`, margin.left, 14);
    }

    function renderMatches() {
      const k = Math.max(1, Math.min(100, Number($('topK').value) || 20));
      const rows = selectedPair.topk.slice(0, k);
      $('matches').innerHTML = rows.map(m => `
        <div class="match">
          <div class="match-head">
            <span>#${m.rank} · score ${score(m.score)} · A[${m.i}] ↔ B[${m.j}]</span>
            <span>${selectedPair.pair_id}</span>
          </div>
          <div class="sentences">
            <div class="sentence">${escapeHtml(m.sentence_a)}</div>
            <div class="sentence">${escapeHtml(m.sentence_b)}</div>
          </div>
        </div>`).join('');
    }

    function renderDocs() {
      $('docsA').innerHTML = docsTable(DATA.documentsA);
      $('docsB').innerHTML = docsTable(DATA.documentsB);
    }

    function docsTable(rows) {
      const maxSent = Math.max(...rows.map(r => r.sentence_count));
      return `<table><tr><th>ID</th><th>File</th><th>Sentences</th></tr>${rows.map(r => `
        <tr><td>${r.doc_id}</td><td>${escapeHtml(r.short_name)}<div class="note">${escapeHtml(r.embeddings_npy)}</div></td><td>${fmt.format(r.sentence_count)}<div class="bar"><span style="width:${100 * r.sentence_count / maxSent}%"></span></div></td></tr>
      `).join('')}</table>`;
    }

    function drawGpu() {
      const canvas = $('gpuCanvas');
      const ctx = canvas.getContext('2d');
      const rows = DATA.gpu;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!rows.length) return;
      const pad = 28, w = canvas.width - 2 * pad, h = canvas.height - 2 * pad;
      ctx.strokeStyle = '#d8d8d8';
      ctx.strokeRect(pad, pad, w, h);
      drawLine(rows.map(r => r.gpu_util_pct), 100, '#a6422a');
      drawLine(rows.map(r => r.mem_used_mib), rows[0].mem_total_mib, '#245c63');
      ctx.fillStyle = '#333';
      ctx.font = '12px system-ui';
      ctx.fillText('GPU util', pad, 16);
      ctx.fillStyle = '#245c63';
      ctx.fillText('memory', pad + 70, 16);
      function drawLine(values, max, color) {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        values.forEach((v, i) => {
          const x = pad + (i / Math.max(1, values.length - 1)) * w;
          const y = pad + h - (v / max) * h;
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      }
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));
    }

    function formatValue(v) {
      return typeof v === 'number' && Number.isFinite(v) ? fmt.format(v) : escapeHtml(v);
    }

    init();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
