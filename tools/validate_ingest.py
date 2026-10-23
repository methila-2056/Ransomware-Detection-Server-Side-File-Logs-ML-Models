"""Ingestion validation report.

Generates ``results/ingest_validation.json`` by running every parser over lines
written to its documented format and through a replay of a generated trace.

The report is explicit about what it does and does not establish. Every parser is
exercised against **synthetic lines**. No capture from a running Samba server,
auditd host or Windows endpoint exists in this repository, so this demonstrates
that the parsers implement the documented formats -- it does not demonstrate that
they survive production traffic, where field ordering, encoding, truncation and
interleaving from multiple hosts all differ from a clean sample.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.ingest import (detect_format, events_to_windows, parse_auditd,
                       parse_samba, parse_windows_events)

SAMPLE_SAMBA = [
    'Feb  1 20:59:12 z1 smbd_audit[35482]: '
    'AD\\alice|10.0.0.5|Data|create_file|ok|/srv/share/a.docx',
    'Feb  1 20:59:13 z1 smbd_audit[35482]: '
    'AD\\alice|10.0.0.5|Data|create_file|ok|/srv/share/b.docx',
    'Feb  1 20:59:14 z1 smbd_audit[35482]: '
    'AD\\alice|10.0.0.5|Data|renameat|ok|/srv/share/a.docx|/srv/share/a.locked',
    'Feb  1 20:59:15 z1 smbd_audit[35482]: '
    'AD\\alice|10.0.0.5|Data|unlinkat|ok|/srv/share/a.locked',
    'Feb  1 20:59:16 z1 smbd_audit[35482]: '
    'AD\\alice|10.0.0.5|Data|open|ok|/srv/share/b.docx',
]

SAMPLE_AUDITD = [
    'type=SYSCALL msg=audit(1717171717.100:456): syscall=257 '
    'exe="/usr/sbin/smbd" name="/srv/share/a.docx" flags=O_CREAT',
    'type=SYSCALL msg=audit(1717171717.200:457): syscall=316 '
    'exe="/usr/sbin/smbd" name="/srv/share/a.docx|/srv/share/a.locked"',
    'type=SYSCALL msg=audit(1717171717.300:458): syscall=263 '
    'exe="/usr/sbin/smbd" name="/srv/share/a.lockx"',
    'type=SYSCALL msg=audit(1717171717.400:459): syscall=0 '
    'exe="/usr/sbin/smbd" name="/srv/share/b.docx"',
    'type=SYSCALL msg=audit(1717171717.500:460): syscall=257 '
    'exe="/bin/sh" name="/tmp/other"',
]

SAMPLE_WINDOWS = [
    {"EventID": 11, "UtcTime": "2026-03-23T10:15:01.000000Z",
     "TargetFilename": "C:\\share\\a.docx", "User": "alice"},
    {"EventID": 11, "UtcTime": "2026-03-23T10:15:02.000000Z",
     "TargetFilename": "C:\\share\\b.docx", "User": "alice"},
    {"EventID": 4663, "AccessMask": "0x00010000",
     "UtcTime": "2026-03-23T10:15:03.000000Z", "TargetFilename": "C:\\share\\b.docx"},
    {"EventID": 23, "UtcTime": "2026-03-23T10:15:04.000000Z",
     "TargetFilename": "C:\\share\\b.docx"},
    {"EventID": 1, "UtcTime": "2026-03-23T10:15:05.000000Z",
     "Image": "C:\\evil.exe"},
]


def parser_entry(name, parser, samples, expected_format=None):
    started = time.perf_counter()
    events, report = parser(samples)
    elapsed = time.perf_counter() - started

    windows = events_to_windows(events, window_seconds=1.0)
    schema_columns = ["window_index", "create_count", "rename_count",
                      "delete_count", "label", "episode_id", "family",
                      "is_stealth", "benign_template"]

    detected = detect_format(samples)
    # detect_format returns a key of ml.ingest.PARSERS, which is not the same
    # string as the parser's display name. Deriving the expectation by splitting
    # the display name on "_" produced "linux" for linux_auditd and so reported
    # the auditd parser as mis-detected when it was detected correctly.
    if expected_format is None:
        expected_format = {"samba_full_audit": "samba",
                           "linux_auditd": "auditd",
                           "windows_events": "windows"}.get(name)

    return {
        "Parser": report.parser,
        "Sample Lines": len(samples),
        "Parsed": report.parsed,
        "Skipped": report.skipped,
        "Parse Rate": round(report.parsed / max(1, report.total_lines), 6),
        "Operations": report.operations,
        "Events Emitted": len(events),
        "Windows Produced": int(len(windows)),
        "Schema Complete": all(c in windows.columns for c in schema_columns),
        "Counts Non Negative": bool(
            all(int(windows[c].min()) >= 0
                for c in ("create_count", "rename_count", "delete_count"))
        ) if len(windows) else True,
        "Parse Milliseconds": round(elapsed * 1000, 3),
        "Detection": detected,
        "Expected Format": expected_format,
        "Format Detected Correctly": (
            detected == expected_format if expected_format else None),
    }


def replay_throughput(n_events=200_000, window_seconds=1.0):
    """Measure parse + aggregate throughput on a large synthetic stream."""
    lines = []
    for i in range(n_events):
        op = ("create_file", "renameat", "unlinkat")[i % 3]
        second = i // 50
        minute = second // 60
        hour = 8 + minute // 60
        stamp = f"{hour:02d}:{minute % 60:02d}:{second % 60:02d}"
        lines.append(f"Feb  1 {stamp} z1 smbd_audit[1]: "
                     f"user{i % 20}|10.0.0.1|Data|{op}|ok|/srv/share/f{i}.docx")

    started = time.perf_counter()
    events, report = parse_samba(lines)
    parse_seconds = time.perf_counter() - started

    started = time.perf_counter()
    windows = events_to_windows(events, window_seconds=window_seconds)
    aggregate_seconds = time.perf_counter() - started

    total = parse_seconds + aggregate_seconds
    return {
        "Events": n_events,
        "Parse Seconds": round(parse_seconds, 3),
        "Aggregate Seconds": round(aggregate_seconds, 3),
        "Total Seconds": round(total, 3),
        "Events Per Second": round(n_events / total) if total else 0,
        "Parsed": report.parsed,
        "Windows": int(len(windows)),
        "Realtime Headroom At 1Hz": round(n_events / total) if total else 0,
    }


def main():
    entries = [
        parser_entry("samba_full_audit", parse_samba, SAMPLE_SAMBA),
        parser_entry("linux_auditd", parse_auditd, SAMPLE_AUDITD),
        parser_entry("windows_events", parse_windows_events, SAMPLE_WINDOWS),
    ]

    throughput = replay_throughput()

    payload = {
        "Purpose": "Verify that the log parsers implement the documented formats "
                   "of the servers the base paper used, and measure their "
                   "throughput.",
        "Formats Supported": [
            "Samba vfs_full_audit (PREFIX|OPERATION|RESULT|FILE)",
            "Linux auditd (type=SYSCALL)",
            "Windows Security 4663 access masks",
            "Sysmon event IDs 1/2/11/23",
        ],
        "Parsers": entries,
        "Throughput": throughput,
        "Validation Status": {
            "Tested Against": "synthetic lines written to each format's "
                              "documented syntax",
            "Not Tested Against": "live capture from a running Samba server, "
                                  "auditd host or Windows endpoint",
            "Known Gaps": [
                "syslog year is absent in RFC3164 and is inferred as the current "
                "year, so logs spanning 31 December parse with a one-year jump",
                "multi-host interleaving assumes monotonic timestamps within a "
                "source; a clock skew between hosts can order windows wrongly",
                "non-UTF8 paths and filenames with embedded pipes are not decoded",
                "truncated or rotated log lines are skipped rather than "
                "partially recovered",
            ],
            "Conclusion": "The parsers implement the documented formats and "
                          "produce the same window schema as the trace "
                          "generator, so real and emulated data share one "
                          "pipeline. They have not been validated against "
                          "production traffic and should be treated as "
                          "unproven against real logs until they are.",
        },
        "Base Paper Comparison": {
            "Paper": "Fluent Bit -> InfluxDB -> CSV -> one-second aggregation "
                     "(Sect. 3.2, 4.2)",
            "This Project": "watchdog observer for live monitoring; "
                            "ml.ingest for offline replay of Samba, auditd, "
                            "Windows and Sysmon logs into the same window "
                            "schema",
            "Still Absent": "an InfluxDB time-series backend and a Fluent Bit "
                            "forwarder; see docker/smb-testbed for a "
                            "provisioning scaffold that was authored but not "
                            "executed in this environment",
        },
    }

    with open("results/ingest_validation.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"{'parser':<22}{'lines':>7}{'parsed':>8}{'skipped':>9}"
          f"{'events':>8}{'schema':>9}{'ms':>8}")
    for entry in entries:
        print(f"{entry['Parser']:<22}{entry['Sample Lines']:>7}"
              f"{entry['Parsed']:>8}{entry['Skipped']:>9}"
              f"{entry['Events Emitted']:>8}"
              f"{str(entry['Schema Complete']):>9}"
              f"{entry['Parse Milliseconds']:>8.2f}")

    t = throughput
    print()
    print(f"throughput: {t['Events']:,} events in {t['Total Seconds']}s "
          f"= {t['Events Per Second']:,}/s "
          f"({t['Parse Seconds']}s parse, {t['Aggregate Seconds']}s aggregate)")

    print()
    print("written to results/ingest_validation.json")


if __name__ == "__main__":
    main()