# SPDX-License-Identifier: Apache-2.0
"""Per-op / per-kernel cost of one decode step, from a Kineto HPU trace.

The V4.1 batch runner wraps every batched decode step in a
``torch.profiler.record_function`` labelled
``v41::request_batch::PP<pp>::generation<g>::requests<n>``.  Those CPU annotations
are the only step boundaries that exist in the trace, so this tool uses them to
partition the *device* timeline: a step owns the device events from its own
submission boundary up to the next submission boundary.  That attributes work
the host submitted for step k to step k, which is the question ("what does one
decode step cost") rather than "what was on the device at instant t".

Two input forms are accepted, because a real HPU trace of a batched decode step
runs to millions of events and must not be ``json.load``-ed:

* a directory holding ``tools/extract_deepseek_v41_trace.py``'s streaming output
  (``inventory.json`` + ``hardware.jsonl.gz`` + ``host.jsonl.gz``) -- preferred;
* a raw chrome trace (``.json`` / ``.json.gz``) -- fine for small captures.

Two independent axes are reported because they answer different questions:

* ``engine``  -- from ``args["HW event name"]``, the runtime's own label
  (TPC_SPU_START_TO_SPU_HALT / MMEH_WB*_MON_TS_BIT1 / DBG_DMA_TRC_*).
* ``op``      -- the event name plus ``args["EventName"]`` and ``recipeName``,
  which name the compiled node and the recipe the kernel came from.

Durations are reported as **unions** (merged, non-overlapping coverage) as well
as sums, and the two are labelled.  Sums are inflated by per-TPC replication
(a TPC kernel records one event per TPC) and, for MME/DMA, by marker
self-overlap; the union is the occupancy figure.  See the Gaudi 2 optimization
manual, ch.1 section 1.8, and ch.3 section 3.6.4 for why the union is not a
removal price either.

Usage:
    python analyze_deepseek_v41_step_profile.py SOURCE [SOURCE ...] \
        [--top 25] [--json OUT] [--assert-no-compile]

    SOURCE may be a trace file or an extractor output directory.
"""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path
import statistics
import sys

STEP_PREFIX = "v41::request_batch::"
DEVICE_CATS = ("kernel", "hpu_op")
ENGINES = ("TPC", "MME", "DMA", "NIC")


def engine_from_marker(marker):
    if not marker:
        return None
    upper = marker.upper()
    for engine in ENGINES:
        if engine in upper:
            return engine
    return upper.split("_")[0] or None


def label(name, event_name, recipe):
    parts = [name]
    if event_name and event_name not in name:
        parts.append(f"[{event_name}]")
    if recipe and recipe not in name:
        parts.append(f"@{recipe}")
    return " ".join(parts)


# --------------------------------------------------------------------------- inputs


def from_extraction(directory):
    """Streaming extractor output. Never materialises the whole trace."""
    inventory = json.loads((directory / "inventory.json").read_text())
    nodes = inventory["nodes"]
    device = []
    with gzip.open(directory / "hardware.jsonl.gz", "rt") as stream:
        for line in stream:
            ts, dur, _lane, node_id, _hw_kind, _api = json.loads(line)
            node = nodes[node_id]
            recipe = str(node.get("recipe", "")).split(":")[0]
            device.append((ts, dur, label(node["kernel"], node.get("node", ""), recipe), node["engine"]))
    cpu = []
    with gzip.open(directory / "host.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            cpu.append((row[0], row[1], row[5]))
    markers = inventory.get("cpu_markers", [])
    return device, cpu, markers, inventory


def from_trace(path):
    """Raw chrome trace. Only for captures small enough to hold in memory."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        document = json.load(stream)
    events = document.get("traceEvents", document)
    device, cpu, markers = [], [], []
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        args = event.get("args") or {}
        duration = event.get("dur", 0)
        if name.startswith(STEP_PREFIX):
            markers.append([event["ts"], duration, name])
            continue
        if duration <= 0:
            continue
        marker = args.get("HW event name")
        if marker or event.get("cat") in DEVICE_CATS:
            recipe = str(args.get("recipeName") or args.get("recipeId") or "")
            device.append((event["ts"], duration, label(name, args.get("EventName", ""), recipe),
                           engine_from_marker(marker)))
        elif event.get("cat") in ("cpu_op", "user_annotation"):
            cpu.append((event["ts"], duration, name))
    return device, cpu, markers, {"source": str(path)}


def load(source):
    if source.is_dir():
        return from_extraction(source)
    return from_trace(source)


# --------------------------------------------------------------------------- analysis


def union(intervals):
    total, left, right = 0.0, None, None
    for start, end in sorted(intervals):
        if right is not None and start <= right:
            right = max(right, end)
        else:
            if right is not None:
                total += right - left
            left, right = start, end
    return total + (right - left if right is not None else 0.0)


def windows_from(markers):
    marks = sorted((mark for mark in markers if str(mark[2]).startswith(STEP_PREFIX)), key=lambda m: m[0])
    windows = []
    for index, mark in enumerate(marks):
        tail = str(mark[2]).rsplit("::", 1)[-1]
        if not tail.startswith("requests"):
            continue
        try:
            requests = int(tail[len("requests"):])
        except ValueError:
            continue
        end = marks[index + 1][0] if index + 1 < len(marks) else mark[0] + mark[1]
        windows.append({"ts": mark[0], "end": end, "dur": mark[1], "requests": requests, "name": str(mark[2])})
    return windows


def is_compile(name):
    return "compilegraph" in str(name).lower().replace("_", "")


def profile(source, top):
    device, cpu, markers, inventory = load(source)
    windows = windows_from(markers)
    report = {
        "source": str(source),
        "device_events": len(device),
        "cpu_events": len(cpu),
        "step_annotations": len(windows),
        "compile_events_total": sum(1 for name in (e[2] for e in cpu) if is_compile(name)),
        "buckets": {},
    }
    if not windows:
        report["error"] = ("no v41::request_batch step annotations: the profile did not cover decode "
                          "(is runner.trace_enabled set? it requires an active profiler)")
        return report
    starts = [window["ts"] for window in windows]
    per_bucket = collections.defaultdict(lambda: {
        "steps": 0, "period_us": [], "annotation_us": [], "compile": 0,
        "device": collections.defaultdict(list), "engine": collections.defaultdict(list),
        "cpu": collections.defaultdict(list),
    })
    for window in windows:
        bucket = per_bucket[window["requests"]]
        bucket["steps"] += 1
        bucket["period_us"].append(window["end"] - window["ts"])
        bucket["annotation_us"].append(window["dur"])

    def window_index(ts):
        index = bisect.bisect_right(starts, ts) - 1
        if index < 0 or ts >= windows[index]["end"]:
            return None
        return index

    for ts, duration, name, engine in device:
        index = window_index(ts)
        if index is None:
            continue
        bucket = per_bucket[windows[index]["requests"]]
        bucket["device"][name].append((ts, ts + duration))
        if engine:
            bucket["engine"][engine].append((ts, ts + duration))
    for ts, duration, name in cpu:
        if name.startswith(STEP_PREFIX):
            continue
        index = window_index(ts)
        if index is None:
            continue
        bucket = per_bucket[windows[index]["requests"]]
        if is_compile(name):
            bucket["compile"] += 1
        bucket["cpu"][name].append((ts, ts + duration))

    for requests in sorted(per_bucket):
        bucket = per_bucket[requests]
        steps = bucket["steps"]
        report["buckets"][requests] = {
            "steps": steps,
            "compile_events_in_window": bucket["compile"],
            "step_period_ms_median": round(statistics.median(bucket["period_us"]) / 1000, 3),
            "annotation_dur_ms_median": round(statistics.median(bucket["annotation_us"]) / 1000, 3),
            "device_union_ms_per_step": round(sum(union(v) for v in bucket["device"].values()) / steps / 1000, 3),
            "device_span_sum_ms_per_step": round(
                sum(sum(b - a for a, b in v) for v in bucket["device"].values()) / steps / 1000, 3),
            "engine_union_ms_per_step": {
                name: round(union(spans) / steps / 1000, 3)
                for name, spans in sorted(bucket["engine"].items(), key=lambda kv: -union(kv[1]))
            },
            "engine_span_sum_ms_per_step": {
                name: round(sum(b - a for a, b in spans) / steps / 1000, 3)
                for name, spans in sorted(bucket["engine"].items(), key=lambda kv: -union(kv[1]))
            },
            "top_device_ops": [{
                "op": name,
                "union_ms_per_step": round(union(spans) / steps / 1000, 4),
                "span_sum_ms_per_step": round(sum(b - a for a, b in spans) / steps / 1000, 4),
                "events_per_step": round(len(spans) / steps, 2),
            } for name, spans in sorted(bucket["device"].items(), key=lambda kv: -union(kv[1]))[:top]],
            "top_cpu_ops": [{
                "op": name,
                "span_sum_ms_per_step": round(sum(b - a for a, b in spans) / steps / 1000, 4),
                "events_per_step": round(len(spans) / steps, 2),
            } for name, spans in sorted(bucket["cpu"].items(),
                                        key=lambda kv: -sum(b - a for a, b in kv[1]))[:top]],
        }
    return report


def bucket_growth(report):
    """Per-op union across every captured bucket, ranked by ABSOLUTE cost at the widest.

    The step time is a step function of the compiled bucket and rows are free
    inside a bucket (measured: n=3 == n=4, n=5..8 all ~46 ms, n=12 == n=16). So a
    slope taken across buckets measures the BUCKET STEP, not a per-row cost, and
    must not be used for ranking. The primary sort here is therefore absolute
    device union at the widest captured bucket -- that is what a per-layer cut has
    to come out of. The adjacent-bucket delta is reported beside it, because
    "what is big" and "what grows with the bucket" are different questions.

    ``events_per_step`` is carried at every bucket so "more launches" and "the
    same launches got slower" stay separable.
    """
    buckets = report.get("buckets", {})
    if len(buckets) < 2:
        return []
    points = sorted(buckets)
    rows = collections.defaultdict(dict)
    for requests, entry in buckets.items():
        for row in entry["top_device_ops"]:
            rows[row["op"]][requests] = row
    result = []
    for op, by_bucket in rows.items():
        widest = points[-1]
        if widest not in by_bucket:
            continue
        narrow = points[-2]
        delta = (by_bucket[widest]["union_ms_per_step"] - by_bucket[narrow]["union_ms_per_step"]
                 if narrow in by_bucket else None)
        result.append({
            "op": op,
            "union_ms_by_bucket": {n: by_bucket[n]["union_ms_per_step"] for n in points if n in by_bucket},
            "events_per_step_by_bucket": {n: by_bucket[n]["events_per_step"] for n in points if n in by_bucket},
            "union_ms_at_widest": by_bucket[widest]["union_ms_per_step"],
            "delta_from_previous_bucket": None if delta is None else round(delta, 4),
            "widest_bucket": widest,
            "previous_bucket": narrow,
        })
    return sorted(result, key=lambda row: -row["union_ms_at_widest"])


def render_growth(report, top):
    rows = bucket_growth(report)
    if not rows:
        return []
    points = sorted(report["buckets"])
    lines = ["", f"## per-op union by bucket, ranked by ABSOLUTE cost at bucket {points[-1]}",
             "## (the step is a step function of the bucket and rows are free inside one, so",
             "##  this absolute column is the target; the delta column says what grows with the bucket)",
             f"  {'op':<46}" + "".join(f"{'u@'+str(n):>9}" for n in points)
             + f"{'d(last)':>9}{'ev/step@last':>13}"]
    for row in rows[:top]:
        cells = "".join(f"{row['union_ms_by_bucket'].get(n, float('nan')):9.3f}" for n in points)
        delta = row["delta_from_previous_bucket"]
        lines.append(f"  {row['op'][:45]:<46}{cells}"
                     f"{('n/a' if delta is None else f'{delta:+.3f}'):>9}"
                     f"{row['events_per_step_by_bucket'].get(points[-1], float('nan')):>13.1f}")
    covered = sum(row["union_ms_at_widest"] for row in rows)
    total = report["buckets"][points[-1]]["device_union_ms_per_step"]
    lines.append(f"  {'SUM over listed ops':<46}{'':>{9*len(points)}}{'':>9}{covered:>13.3f}")
    lines.append(f"  {'device UNION at bucket '+str(points[-1]):<46}{'':>{9*len(points)}}{'':>9}{total:>13.3f}")
    return lines


def render(report):
    lines = [f"# {report['source']}",
             f"# device events {report['device_events']}  cpu events {report['cpu_events']}"
             f"  step annotations {report['step_annotations']}"
             f"  compileGraph events {report.get('compile_events_total', 0)}"]
    if "error" in report:
        return "\n".join(lines + [f"ERROR: {report['error']}"])
    for requests, entry in report["buckets"].items():
        lines += ["", f"## bucket {requests}  ({entry['steps']} steps)"]
        lines.append(f"compile events in window   {entry['compile_events_in_window']:8d}"
                     + ("   <-- NOT WARM, profile void" if entry["compile_events_in_window"] else "   (warm)"))
        lines.append(f"step period (submit-to-submit, median) {entry['step_period_ms_median']:8.3f} ms")
        lines.append(f"host annotation dur (median)           {entry['annotation_dur_ms_median']:8.3f} ms")
        lines.append(f"device UNION / step        {entry['device_union_ms_per_step']:8.3f} ms   <- occupancy")
        lines.append(f"device SUM   / step        {entry['device_span_sum_ms_per_step']:8.3f} ms   (inflated)")
        lines.append("engine union / step: " + ", ".join(
            f"{name} {value:.3f}" for name, value in entry["engine_union_ms_per_step"].items()))
        lines.append("engine sum   / step: " + ", ".join(
            f"{name} {value:.3f}" for name, value in entry["engine_span_sum_ms_per_step"].items())
            + "   (marker figures, not occupancy)")
        lines += ["", f"  {'op':<58}{'union ms':>10}{'sum ms':>10}{'ev/step':>9}"]
        for row in entry["top_device_ops"]:
            lines.append(f"  {row['op'][:57]:<58}{row['union_ms_per_step']:>10.4f}"
                         f"{row['span_sum_ms_per_step']:>10.4f}{row['events_per_step']:>9.2f}")
        lines += ["", f"  {'cpu op':<58}{'sum ms':>10}{'ev/step':>9}"]
        for row in entry["top_cpu_ops"]:
            lines.append(f"  {row['op'][:57]:<58}{row['span_sum_ms_per_step']:>10.4f}"
                         f"{row['events_per_step']:>9.2f}")
    lines += render_growth(report, 30)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--assert-no-compile", action="store_true",
                        help="Exit nonzero if the acquisition contains compileGraph activity (not warm)")
    args = parser.parse_args()
    reports = [profile(source, args.top) for source in args.sources]
    for report in reports:
        print(render(report))
        print()
    if args.json:
        args.json.write_text(json.dumps(reports, indent=2) + "\n")
    failed = any("error" in report for report in reports)
    if args.assert_no_compile:
        compiled = sum(report.get("compile_events_total", 0) for report in reports)
        if compiled:
            print(f"FAIL: {compiled} compileGraph events in the acquisition; the profile is not warm")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
