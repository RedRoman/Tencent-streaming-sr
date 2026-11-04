"""Classify observed CPU-fallback diagnostics without treating worker prose as logs.

Only runtime stderr, non-worker stdout, and explicit warnings are classified.
Known worker JSON records are masked with offsets preserved; their warnings are
extracted separately. A false result means no observed diagnostic, not proof of
native kernel execution. No accelerator libraries are imported.
"""
from __future__ import annotations

import json
import re
from typing import Sequence


_FALLBACK = re.compile(
    r"\b(?:fall(?:s|ing)?[ \t_-]*back|fallback)\b[^\n.!?]{0,180}?\bCPU\b"
    r"|\bCPU[ \t_-]+(?:fall[ \t_-]*back|fallback)\b", re.I)
_NEGATED = re.compile(
    r"\b(?:no|without)(?:[ \t]+\w+){0,3}[ \t]+CPU[ \t_-]+fallback\b"
    r"|\b(?:no|without)(?:[ \t]+\w+){0,3}[ \t]+fallback(?:[ \t]+\w+){0,4}[ \t]+CPU\b"
    r"|\b(?:not|never)[ \t]+(?:falling|falls?|fall)[ \t_-]*back(?:[ \t]+\w+){0,4}[ \t]+CPU\b"
    r"|\bnot[ \t]+(?:using|used|enabling|invoking)[ \t]+(?:the[ \t]+)?CPU[ \t_-]+fallback\b"
    r"|\bCPU[ \t_-]+fallback[ \t]+(?:(?:was|is)[ \t]+)?(?:not[ \t]+(?:observed|detected|used)|disabled|absent)\b",
    re.I)
_OPERATOR = re.compile(r"\b(?:The\s+)?operator\s+['\"]([^'\"]+)['\"]", re.I)


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _worker_record(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if set(value) == {"worker_final"}:
        return _worker_record(value["worker_final"])
    if set(value) == {"step"} and isinstance(value["step"], dict):
        return {"name", "status"} <= value["step"].keys()
    return {"case", "pid", "physical_card", "steps", "native_evidence_scope",
            "model_or_checkpoint_loaded"} <= value.keys() and isinstance(value["steps"], list)


def _mask_worker_stdout(stdout: str):
    decoder = json.JSONDecoder()
    excluded, warnings = [], []
    covered_until = 0
    masked = list(stdout)
    for start in re.finditer(r"(?m)^[ \t]*\{", stdout):
        begin = stdout.index("{", start.start(), start.end())
        if begin < covered_until:
            continue
        try:
            value, end = decoder.raw_decode(stdout, begin)
        except json.JSONDecodeError:
            continue
        if not _worker_record(value):
            continue
        covered_until = end
        record_index = len(excluded)
        excluded.append({"source": "stdout", "span": [begin, end],
                         "reason": "worker-generated JSON; explicit warnings classified separately"})
        for index in range(begin, end):
            if masked[index] != "\n":
                masked[index] = " "
        payload = value.get("worker_final", value)
        prefix = "worker_final." if "worker_final" in value else ""
        steps = [(prefix + "step", payload["step"])] if "step" in payload else [
            (f"{prefix}steps[{index}]", step) for index, step in enumerate(payload["steps"])]
        for step_path, step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("warnings", []), list):
                continue
            for warning_index, warning in enumerate(step.get("warnings", [])):
                if isinstance(warning, str):
                    warnings.append((f"stdout.worker_json[{record_index}].{step_path}.warnings[{warning_index}]",
                                     warning, [begin, end]))
    return "".join(masked), excluded, warnings


def _classify_source(source: str, text: str, raw: str, container_span=None):
    negated = list(_NEGATED.finditer(text))
    positives, negatives = [], []
    for candidate in _FALLBACK.finditer(text):
        covering = next((item for item in negated if _overlaps(item.span(), candidate.span())), None)
        match = covering or candidate
        entry = {"source": source, "span": list(match.span()), "matched_text": raw[match.start():match.end()]}
        if container_span is not None:
            entry["stdout_worker_json_container_span"] = container_span
            entry["span_basis"] = "decoded warning string"
        else:
            entry["span_basis"] = "original source string"
        if covering:
            if entry not in negatives:
                negatives.append(entry)
            continue
        line_begin = raw.rfind("\n", 0, candidate.start()) + 1
        line_end = raw.find("\n", candidate.end())
        if line_end < 0:
            line_end = len(raw)
        preceding_operators = list(_OPERATOR.finditer(raw[line_begin:candidate.start()]))
        operator = preceding_operators[-1] if preceding_operators else None
        if operator:
            entry["operator"] = operator.group(1)
            sentence_begin = line_begin + operator.start()
            sentence_end = raw.find(".", candidate.end(), line_end)
            sentence_end = sentence_end + 1 if sentence_end >= 0 else candidate.end()
            entry["operator_diagnostic_span"] = [sentence_begin, sentence_end]
            entry["operator_diagnostic_text"] = raw[sentence_begin:sentence_end]
        positives.append(entry)
    return positives, negatives


def classify_cpu_fallback_logs(*, stdout: str = "", stderr: str = "",
                               warnings: Sequence[str] = ()) -> dict:
    """Return positive diagnostics and exclusions with zero-based half-open spans.

    Spans use Unicode character offsets in the named source (not byte offsets).
    Structured warning spans refer to decoded warning text, with its raw JSON
    container span also supplied. Worker return codes remain the caller's job.
    """
    masked, excluded, embedded = _mask_worker_stdout(stdout)
    sources = [("stderr", stderr, stderr, None), ("stdout", masked, stdout, None)]
    sources.extend((f"warnings[{index}]", warning, warning, None)
                   for index, warning in enumerate(warnings))
    sources.extend((name, warning, warning, container) for name, warning, container in embedded)
    diagnostics, negated = [], []
    for name, text, raw, container in sources:
        positive, negative = _classify_source(name, text, raw, container)
        diagnostics.extend(positive)
        negated.extend(negative)
    return {"cpu_fallback_observed": bool(diagnostics), "diagnostics": diagnostics,
            "negated_mentions": negated, "excluded_worker_json": excluded,
            "interpretation": "No positive diagnostic means no observed CPU fallback; kernel placement is not proved",
            "process_exit_status_evaluated": False}
