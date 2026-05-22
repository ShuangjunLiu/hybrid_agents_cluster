#!/usr/bin/env python3
"""Aggregate MBPP single-file worker summary.json artifacts."""

import argparse
import json
import os
import statistics
import sys


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def find_summaries(root):
    matches = []
    for directory, _, filenames in os.walk(root):
        if "summary.json" in filenames:
            matches.append(os.path.join(directory, "summary.json"))
    return sorted(matches)


def task_id_from_summary(summary, summary_path):
    benchmark = summary.get("benchmark") or {}
    if benchmark.get("task_id") is not None:
        return int(benchmark["task_id"])
    parts = summary_path.split(os.sep)
    for part in parts:
        if part.startswith("task_"):
            suffix = part[len("task_") :]
            if suffix.isdigit():
                return int(suffix)
    return None


def worker_duration(summary):
    generation = summary.get("generation") or {}
    if generation.get("duration_seconds") is not None:
        return float(generation["duration_seconds"])
    qwen = summary.get("qwen") or {}
    duration = qwen.get("duration_seconds")
    return float(duration) if duration is not None else None


def classify_counts(summaries):
    counts = {
        "timeout_count": 0,
        "empty_patch_count": 0,
        "qwen_failure_count": 0,
        "endpoint_failure_count": 0,
        "invalid_model_output_count": 0,
        "tests_failed_count": 0,
    }
    for summary in summaries:
        failure_class = summary.get("failure_class")
        if failure_class == "qwen_timeout":
            counts["timeout_count"] += 1
        if failure_class == "empty_patch":
            counts["empty_patch_count"] += 1
        if failure_class == "qwen_failed":
            counts["qwen_failure_count"] += 1
        if failure_class in ("setup_error", "endpoint_failure"):
            counts["endpoint_failure_count"] += 1
        if failure_class == "invalid_model_output":
            counts["invalid_model_output_count"] += 1
        if failure_class == "tests_failed":
            counts["tests_failed_count"] += 1
    return counts


def aggregate(root):
    summary_paths = find_summaries(root)
    summaries = [read_json(path) for path in summary_paths]
    total = len(summaries)
    pass_count = sum(1 for summary in summaries if summary.get("ok") is True)
    durations = [
        duration
        for duration in (worker_duration(summary) for summary in summaries)
        if duration is not None
    ]
    tasks = []
    for path, summary in zip(summary_paths, summaries):
        tasks.append(
            {
                "task_id": task_id_from_summary(summary, path),
                "ok": summary.get("ok") is True,
                "failure_class": summary.get("failure_class"),
                "duration_seconds": worker_duration(summary),
                "artifact_dir": summary.get("artifact_dir") or os.path.dirname(path),
                "changed_paths": summary.get("changed_paths") or [],
                "disallowed_paths": summary.get("disallowed_paths") or [],
            }
        )
    tasks.sort(key=lambda item: (item["task_id"] is None, item["task_id"] or 0, item["artifact_dir"]))
    result = {
        "schema_version": 1,
        "benchmark": "mbpp_sanitized_single_file",
        "run_root": os.path.abspath(root),
        "total_tasks": total,
        "pass_count": pass_count,
        "pass_rate": round(pass_count / total, 4) if total else None,
        "median_runtime_seconds": statistics.median(durations) if durations else None,
        "tasks": tasks,
    }
    result.update(classify_counts(summaries))
    return result


def print_text(result):
    print("run_root: {}".format(result["run_root"]))
    print("total_tasks: {}".format(result["total_tasks"]))
    print("pass_count: {}".format(result["pass_count"]))
    print("pass_rate: {}".format(result["pass_rate"]))
    print("timeout_count: {}".format(result["timeout_count"]))
    print("empty_patch_count: {}".format(result["empty_patch_count"]))
    print("qwen_failure_count: {}".format(result["qwen_failure_count"]))
    print("endpoint_failure_count: {}".format(result["endpoint_failure_count"]))
    print("invalid_model_output_count: {}".format(result["invalid_model_output_count"]))
    print("tests_failed_count: {}".format(result["tests_failed_count"]))
    print("median_runtime_seconds: {}".format(result["median_runtime_seconds"]))
    print("tasks:")
    for task in result["tasks"]:
        print(
            "- task_id={task_id} ok={ok} failure_class={failure_class} "
            "duration={duration_seconds} artifact_dir={artifact_dir}".format(**task)
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", help="Run directory or artifact root to scan recursively.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    args = parser.parse_args(argv)

    result = aggregate(args.run_root)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_text(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
