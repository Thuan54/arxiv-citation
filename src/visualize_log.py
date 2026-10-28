"""Turn crawler PIPELINE_HEALTH log entries into a standalone HTML dashboard.

Usage: python3 visualize_log.py crawler.log
"""

import argparse
import json
import re
import sys
from pathlib import Path


HEALTH_MARKER = "PIPELINE_HEALTH "
TIMESTAMP_PATTERN = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)")


def read_snapshots(log_path):
    snapshots = []
    malformed = 0

    with log_path.open("r", encoding="utf-8", errors="replace") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            marker_index = line.find(HEALTH_MARKER)
            if marker_index < 0:
                continue

            payload = line[marker_index + len(HEALTH_MARKER):].strip()
            try:
                snapshot = json.loads(payload)
            except json.JSONDecodeError:
                malformed += 1
                continue

            timestamp_match = TIMESTAMP_PATTERN.match(line)
            timestamp = (
                timestamp_match.group(1)
                if timestamp_match
                else f"Snapshot {len(snapshots) + 1}"
            )
            snapshots.append({
                "timestamp": timestamp,
                "line": line_number,
                "data": snapshot,
            })

    return snapshots, malformed


def render_dashboard(snapshots, source_name):
    data = json.dumps(snapshots, ensure_ascii=False).replace("<", "\\u003c")
    source = json.dumps(source_name, ensure_ascii=False).replace(
        "<", "\\u003c"
    )
    page = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Crawler health dashboard</title>
<style>
:root { color-scheme: dark; font: 15px/1.5 system-ui, sans-serif; background: #10151d; color: #e8edf5; }
body { max-width: 1200px; margin: 0 auto; padding: 28px 20px 50px; }
h1 { margin-bottom: 4px; font-size: 1.8rem; }
.muted { color: #9aa8bb; }
.queue-summary { min-height: 24px; font-size: .85rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 16px; margin-top: 20px; }
.card { background: #19212c; border: 1px solid #293647; border-radius: 12px; padding: 16px; min-width: 0; }
.card h2 { font-size: 1rem; margin: 0 0 12px; }
.wide { grid-column: 1 / -1; }
canvas { width: 100%; height: 245px; display: block; }
.workers { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 10px; }
.worker { border: 1px solid #304052; background: #141b24; border-radius: 8px; padding: 11px; }
.worker strong { display: block; overflow-wrap: anywhere; }
.worker .state { color: #87c6ff; font-weight: 650; }
.worker .task { color: #c3cedb; overflow-wrap: anywhere; }
.worker .duration { color: #ffd479; font-variant-numeric: tabular-nums; }
.empty { color: #9aa8bb; padding: 18px 0; }
@media (max-width: 600px) { body { padding: 18px 12px 36px; } canvas { height: 210px; } }
</style>
</head>
<body>
<h1>Crawler health</h1>
<div class="muted" id="summary"></div>
<div class="grid">
  <section class="card"><h2>Queue pressure</h2><div class="muted queue-summary" id="queue-summary"></div><canvas id="queues"></canvas></section>
  <section class="card"><h2>Completed work</h2><canvas id="completed"></canvas></section>
  <section class="card"><h2>Time since completion</h2><canvas id="stalled"></canvas></section>
  <section class="card wide"><h2>Latest worker states</h2><div class="workers" id="workers"></div></section>
</div>
<script>
const snapshots = __DATA__;
const colors = ["#55c2a3", "#69a9ff", "#ffb454", "#e879f9", "#f27783", "#c4a7ff"];
const summary = document.getElementById("summary");
const latest = snapshots[snapshots.length - 1];
summary.textContent = `${snapshots.length} health snapshots from ${__SOURCE__} · Latest: ${latest.timestamp}`;

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  return {ctx, width: rect.width, height: rect.height};
}

function drawLines(canvasId, series, unit) {
  const {ctx, width, height} = setupCanvas(document.getElementById(canvasId));
  const left = 43, right = 12, top = 14, bottom = 50;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const values = series.flatMap(item => item.values).filter(Number.isFinite);
  const maxValue = Math.max(1, ...values);
  ctx.font = "11px system-ui";
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#344252";
  ctx.fillStyle = "#9aa8bb";
  for (let tick = 0; tick <= 4; tick++) {
    const value = maxValue * tick / 4;
    const y = top + plotHeight * (1 - tick / 4);
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width - right, y); ctx.stroke();
    ctx.fillText(`${value.toFixed(value < 10 ? 1 : 0)}${unit}`, 2, y + 4);
  }
  series.forEach((item, index) => {
    if (!item.values.length) return;
    ctx.strokeStyle = colors[index % colors.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    item.values.forEach((value, point) => {
      const x = left + (series[0].values.length <= 1 ? 0 : point * plotWidth / (series[0].values.length - 1));
      const y = top + plotHeight * (1 - value / maxValue);
      if (point === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  const legendY = height - 8;
  let legendX = left;
  series.forEach((item, index) => {
    if (!item.values.length) return;
    ctx.fillStyle = colors[index % colors.length];
    ctx.fillRect(legendX, legendY - 9, 9, 9);
    ctx.fillStyle = "#c3cedb";
    ctx.fillText(item.name, legendX + 14, legendY);
    legendX += 20 + ctx.measureText(item.name).width;
  });
  if (snapshots.length) {
    const axisY = top + plotHeight + 16;
    ctx.fillStyle = "#9aa8bb";
    ctx.textAlign = "left";
    ctx.fillText(snapshots[0].timestamp, left, axisY);
    ctx.textAlign = "right";
    ctx.fillText(snapshots[snapshots.length - 1].timestamp, width - right, axisY);
    ctx.textAlign = "left";
  }
}

const queueNames = [...new Set(snapshots.flatMap(s => Object.keys(s.data.queues || {})))];
const queueSeries = queueNames.map(name => ({
  name,
  values: snapshots.map(s => {
    const q = (s.data.queues || {})[name];
    return q && q.capacity ? 100 * q.size / q.capacity : 0;
  })
}));
document.getElementById("queue-summary").textContent = queueNames
  .map(name => {
    const queue = (latest.data.queues || {})[name];
    return `${name}: ${queue.size}/${queue.capacity}`;
  })
  .join(" · ");
drawLines("queues", queueSeries, "%");

const completionNames = [...new Set(snapshots.flatMap(s => Object.keys(s.data.completed || {})))];
const completionSeries = completionNames.map(name => ({
  name,
  values: snapshots.map(s => (s.data.completed || {})[name] || 0)
}));
drawLines("completed", completionSeries, "");
drawLines("stalled", [{
  name: "seconds",
  values: snapshots.map(s => Number(s.data.seconds_since_completion) || 0)
}], "s");

const workers = latest.data.workers || {};
const workerRoot = document.getElementById("workers");
const workerEntries = Object.entries(workers);
if (!workerEntries.length) {
  workerRoot.innerHTML = '<div class="empty">No worker state was recorded in the latest snapshot.</div>';
} else {
  for (const [name, worker] of workerEntries) {
    const card = document.createElement("div");
    card.className = "worker";
    const title = document.createElement("strong");
    title.textContent = name;
    const state = document.createElement("div");
    state.className = "state";
    state.textContent = worker.state || "unknown";
    const task = document.createElement("div");
    task.className = "task";
    task.textContent = worker.task ? `Task: ${worker.task}` : "No active task";
    const duration = document.createElement("div");
    duration.className = "duration";
    duration.textContent = `${Number(worker.duration_seconds || 0).toFixed(1)}s in this state`;
    const details = document.createElement("div");
    details.className = "muted";
    details.textContent = Object.entries(worker.details || {}).map(([key, value]) => `${key}: ${value}`).join(" · ");
    card.append(title, state, task, duration, details);
    workerRoot.append(card);
  }
}

window.addEventListener("resize", () => {
  drawLines("queues", queueSeries, "%");
  drawLines("completed", completionSeries, "");
  drawLines("stalled", [{name: "seconds", values: snapshots.map(s => Number(s.data.seconds_since_completion) || 0)}], "s");
});
</script>
</body>
</html>
"""
    page = page.replace("__DATA__", data)
    page = page.replace("__SOURCE__", source)
    return page


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a standalone HTML dashboard from crawler health logs."
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        type=Path,
        default=Path("crawler.log"),
        help="crawler log path (default: crawler.log)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="HTML output path (default: crawler-health.html next to the log)",
    )
    args = parser.parse_args(argv)

    if not args.log_file.is_file():
        parser.error(f"log file not found: {args.log_file}")

    snapshots, malformed = read_snapshots(args.log_file)
    if not snapshots:
        parser.error(
            f"no valid {HEALTH_MARKER.strip()} entries found in {args.log_file}"
        )

    output_path = args.output or args.log_file.with_name("crawler-health.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_dashboard(snapshots, args.log_file.name),
        encoding="utf-8",
    )

    print(f"Wrote {len(snapshots)} health snapshots to {output_path}")
    if malformed:
        print(
            f"Skipped {malformed} malformed health log entr"
            f"{'y' if malformed == 1 else 'ies'}.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
