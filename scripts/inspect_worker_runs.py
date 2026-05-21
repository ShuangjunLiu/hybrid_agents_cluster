#!/usr/bin/env python3
"""Inspect hybrid worker run registry records and summary artifacts."""

import argparse
import json
import os
import sys


DEFAULT_ARTIFACT_ROOT = "/tmp/hybrid_agent_tasks"
DEFAULT_REGISTRY_NAME = "runs.jsonl"


def read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def load_registry(path):
    records = []
    with open(path, "r") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError("{}:{}: invalid JSON: {}".format(path, line_number, exc))
            record["_registry_line"] = line_number
            records.append(record)
    return records


def resolve_registry(args):
    if args.registry:
        return os.path.abspath(args.registry)
    artifact_root = os.path.abspath(args.artifact_root or DEFAULT_ARTIFACT_ROOT)
    return os.path.join(artifact_root, DEFAULT_REGISTRY_NAME)


def record_status(record):
    return "ok" if record.get("ok") is True else "failed"


def filter_records(records, status):
    if status == "all":
        return records
    return [record for record in records if record_status(record) == status]


def limited_recent(records, limit):
    records = list(records)
    if limit is None:
        return records
    if limit <= 0:
        return []
    return records[-limit:]


def format_paths(paths):
    if not paths:
        return "-"
    return ",".join(paths)


def compact_record(record):
    return {
        "timestamp": record.get("timestamp"),
        "status": record_status(record),
        "ok": record.get("ok"),
        "failure_class": record.get("failure_class"),
        "mode": record.get("mode"),
        "model": record.get("model"),
        "changed_paths": record.get("changed_paths") or [],
        "artifact_dir": record.get("artifact_dir"),
    }


def print_records(records):
    if not records:
        print("No worker runs found.")
        return
    for record in records:
        item = compact_record(record)
        print(
            "{timestamp} {status} {failure_class} {mode} {model} {paths} {artifact_dir}".format(
                timestamp=item["timestamp"] or "-",
                status=item["status"],
                failure_class=item["failure_class"] or "-",
                mode=item["mode"] or "-",
                model=item["model"] or "-",
                paths=format_paths(item["changed_paths"]),
                artifact_dir=item["artifact_dir"] or "-",
            )
        )


def command_status(command_summary):
    if not command_summary:
        return "not run"
    status = "ok" if command_summary.get("returncode") == 0 else "failed"
    if command_summary.get("timeout_occurred"):
        status = "timeout"
    return "{} rc={} duration={}s".format(
        status,
        command_summary.get("returncode"),
        command_summary.get("duration_seconds"),
    )


def print_summary(summary):
    print("artifact_dir: {}".format(summary.get("artifact_dir") or "-"))
    print("status: {}".format("ok" if summary.get("ok") is True else "failed"))
    print("failure_class: {}".format(summary.get("failure_class") or "-"))
    print("model: {}".format(summary.get("model") or "-"))
    print("mode: {}".format(summary.get("mode") or "-"))
    print("qwen: {}".format(command_status(summary.get("qwen"))))
    tests = summary.get("tests") or []
    if tests:
        for test in tests:
            print("test: {}".format(command_status(test)))
    else:
        print("tests: none")
    print("changed_paths: {}".format(format_paths(summary.get("changed_paths") or [])))
    print("disallowed_paths: {}".format(format_paths(summary.get("disallowed_paths") or [])))
    print("patch_sha256: {}".format(summary.get("patch_sha256") or "-"))
    reasons = summary.get("failure_reasons") or []
    if reasons:
        print("failure_reasons:")
        for reason in reasons:
            print("- {}".format(reason))
    print("replay_command: {}".format(summary.get("replay_command") or "-"))


def show_summary(artifact_dir, as_json):
    summary_path = os.path.join(os.path.abspath(artifact_dir), "summary.json")
    summary = read_json(summary_path)
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_summary(summary)


def list_runs(args):
    registry = resolve_registry(args)
    records = load_registry(registry)
    records = filter_records(records, args.status)
    records = limited_recent(records, args.limit)
    if args.json:
        print(json.dumps([compact_record(record) for record in records], indent=2, sort_keys=True))
    else:
        print_records(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", help="Artifact root containing runs.jsonl.")
    parser.add_argument("--registry", help="Path to a JSONL run registry.")
    parser.add_argument("--limit", type=int, default=20, help="Number of recent records to show. Use 0 for none.")
    parser.add_argument("--status", choices=["all", "ok", "failed"], default="all")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--show", metavar="ARTIFACT_DIR", help="Show ARTIFACT_DIR/summary.json.")
    args = parser.parse_args(argv)

    try:
        if args.show:
            show_summary(args.show, args.json)
        else:
            list_runs(args)
    except Exception as exc:
        print("inspect_worker_runs: {}".format(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
