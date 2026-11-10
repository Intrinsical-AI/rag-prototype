"""Classify strict security reports; operational failures never become soft passes."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

MAX_BYTES = 8 * 1024 * 1024
MAX_DEPENDENCIES = 1000


def read_text(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > MAX_BYTES:
            raise ValueError("Report/input must be one bounded regular file")
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES:
        raise ValueError("Report/input grew beyond its limit")
    return content.decode("utf-8")


def object_value(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Expected a JSON object with string keys")
    return cast(dict[str, object], value)


def list_value(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list")
    return cast(list[object], value)


def active_requirements(text: str) -> dict[str, Version]:
    """Admit UV's fully pinned/hash export, then select this interpreter's markers."""
    result: dict[str, Version] = {}
    for raw in re.sub(r"\\\r?\n", " ", text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"\s+(?=--hash=)", line)
        requirement = Requirement(parts[0].strip())
        hashes = [part.strip() for part in parts[1:]]
        if not hashes or not all(re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", h) for h in hashes):
            raise ValueError("Every requirement must carry SHA256 hashes")
        specs = list(requirement.specifier)
        if requirement.url or requirement.extras or len(specs) != 1 or specs[0].operator != "==":
            raise ValueError("Expected flattened exact requirements, without URLs or extras")
        version = Version(specs[0].version)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        if name in result and result[name] != version:
            raise ValueError("Conflicting active requirement versions")
        result[name] = version
        if len(result) > MAX_DEPENDENCIES:
            raise ValueError("Too many audited requirements")
    if not result:
        raise ValueError("An empty audit input is not a successful scan")
    return result


@dataclass(frozen=True)
class Outcome:
    status: Literal["clean", "findings", "error"]
    findings: int
    message: str


def consistent(returncode: int, findings: int) -> Outcome:
    if returncode != (1 if findings else 0):
        return Outcome("error", findings, "Tool exit status and complete report disagree")
    return Outcome("findings" if findings else "clean", findings, "Complete report")


def bandit_outcome(report: object, returncode: int) -> Outcome:
    value = object_value(report)
    if list_value(value["errors"]):
        return Outcome("error", 0, "Bandit reported operational errors")
    totals = object_value(object_value(value["metrics"])["_totals"])
    if type(totals["loc"]) is not int or totals["loc"] <= 0:
        raise ValueError("Bandit must report positive scanned lines of code")
    findings = list_value(value["results"])
    for item in findings:
        row = object_value(item)
        if row["issue_severity"] not in {"MEDIUM", "HIGH"} or row["issue_confidence"] not in {
            "MEDIUM",
            "HIGH",
        }:
            raise ValueError("Unexpected Bandit filtering/report")
        if not isinstance(row["issue_text"], str) or not isinstance(row["filename"], str):
            raise ValueError("Malformed Bandit finding")
        if not isinstance(row["test_id"], str) or type(row["line_number"]) is not int:
            raise ValueError("Malformed Bandit finding location")
    return consistent(returncode, len(findings))


def audit_outcome(report: object, returncode: int, expected: dict[str, Version]) -> Outcome:
    value = object_value(report)
    if set(value) != {"dependencies", "fixes"} or list_value(value["fixes"]):
        raise ValueError("Expected the non-remediating pip-audit JSON contract")
    dependencies = list_value(value["dependencies"])
    if len(dependencies) > MAX_DEPENDENCIES:
        raise ValueError("Too many audit response dependencies")
    seen: dict[str, Version] = {}
    count = 0
    for item in dependencies:
        row = object_value(item)
        if (
            "skip_reason" in row
            or not isinstance(row["name"], str)
            or not isinstance(row["version"], str)
        ):
            raise ValueError("Skipped or malformed audit dependency")
        name = canonicalize_name(row["name"])
        if name in seen:
            raise ValueError("Duplicate audit response dependency")
        seen[name] = Version(row["version"])
        vulnerabilities = list_value(row["vulns"])
        for vulnerability in vulnerabilities:
            finding = object_value(vulnerability)
            identity = finding["id"]
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("Malformed vulnerability identifier")
            if not all(isinstance(v, str) for v in list_value(finding["fix_versions"])):
                raise ValueError("Malformed vulnerability fix versions")
        count += len(vulnerabilities)
        if count > 10000:
            raise ValueError("Too many findings")
    if seen != expected:
        raise ValueError("Audit response does not cover the exact active requirement versions")
    return consistent(returncode, count)


def classify(
    path: Path, returncode: int, expected: dict[str, Version] | None
) -> tuple[Outcome, object]:
    try:
        report: object = json.loads(read_text(path))
        outcome = (
            bandit_outcome(report, returncode)
            if expected is None
            else audit_outcome(report, returncode, expected)
        )
        return outcome, report
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        return Outcome("error", 0, "Missing, malformed, partial or skipped security report"), None


def bandit_locations(report: object, outcome: Outcome) -> list[dict[str, object]]:
    """Show actionable locations without copying code or possible secret literals."""
    if outcome.status == "error":
        return []
    rows = list_value(object_value(report)["results"])
    return [
        {
            key: object_value(row)[key]
            for key in ("filename", "line_number", "test_id", "issue_severity", "issue_confidence")
        }
        for row in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", required=True, type=Path)
    parser.add_argument("--check-input", action="store_true")
    parser.add_argument("--bandit-report", type=Path)
    parser.add_argument("--bandit-returncode", type=int)
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--audit-returncode", type=int)
    parser.add_argument("--soft-findings", action="store_true")
    args = parser.parse_args()
    try:
        expected = active_requirements(read_text(args.requirements))
    except (OSError, ValueError):
        print(json.dumps({"status": "error", "message": "Invalid frozen hashed audit input"}))
        return 2
    if args.check_input:
        print(json.dumps({"status": "input_validated", "active_dependencies": len(expected)}))
        return 0
    if any(
        value is None
        for value in (
            args.bandit_report,
            args.bandit_returncode,
            args.audit_report,
            args.audit_returncode,
        )
    ):
        parser.error("Both auditor reports and return codes are required")
    bandit, bandit_report = classify(args.bandit_report, args.bandit_returncode, None)
    audit, audit_report = classify(args.audit_report, args.audit_returncode, expected)
    error = any(result.status == "error" for result in (bandit, audit))
    findings = bandit.findings + audit.findings
    print(
        json.dumps(
            {
                "status": "error" if error else "findings" if findings else "clean",
                "soft_findings": args.soft_findings,
                "bandit": asdict(bandit),
                "dependency_audit": asdict(audit),
                "bandit_locations": bandit_locations(bandit_report, bandit),
                "dependency_report": audit_report if audit.status != "error" else None,
            },
            sort_keys=True,
        )
    )
    return 2 if error else 1 if findings and not args.soft_findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
