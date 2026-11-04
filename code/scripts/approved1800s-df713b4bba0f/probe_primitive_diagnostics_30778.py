"""Prepare one new-authorized <=300s primitive diagnostic phase on 30778.

Default mode is check-only: stdlib/source inspection, no claim or npu-smi and no
Torch/NPU imports. --execute requires an explicit new phase-allocation record/token
and a fresh successful CPU preflight bound to all source hashes/runtime roots.
This phase does not consume or reuse the earlier 120s authorization/claim, and
does not automatically expand into the proposed 30-minute investigation.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from classify_cpu_fallback_logs import classify_cpu_fallback_logs


PROJECT = Path("/home/ma-user/modelarts/user-job-dir/sol_h3")
PHYSICAL_CARD = 7
SCOPE = "sol_h3.primitive_diagnostics_300s_first_stage.v1"
CASES = {"NPU_init_tiny_alloc_free": 20, "SDPA_BF16_mask128": 30,
         "depthwise_Conv2d_BF16_C128_k5": 100, "VAE_Conv3d_FP32master_FP16autocast": 100}
TOTAL_SECONDS, SHARED_SECONDS = 300, 260
OUTER_TERM, OUTER_KILL = 285, 2
MAIN_WAIT, MAIN_REAP, FINAL_IDLE_TIMEOUT = 289, 1, 2
FROZEN_PROBE_SHA = "af1c4e2326a976134399a059b440c7e42c12540269dd82abe74ba31034097a7a"
CLASSIFIER_SHA = "8dcae3980e09d5a0ef724e6573c80a9030fb6af177ba1ddc6b99c822b2b3a491"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temp, path)


def kill_groups(groups):
    errors = []
    for group in list(groups):
        try: os.killpg(group, signal.SIGKILL)
        except ProcessLookupError: pass
        except OSError as error:
            errors.append({"group_id": group, "errno": error.errno, "error": str(error)})
    return errors


def idle_snapshot(timeout):
    result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=timeout)
    return {"exit": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
            "card_idle": result.returncode == 0 and
            f"No running processes found in NPU {PHYSICAL_CARD}" in result.stdout}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Frozen worker transform anchor mismatch")
    return source.replace(old, new, 1)


def build_worker(frozen_path):
    """Reuse checked tiny operations/cleanup, with explicit auditable transforms."""
    if sha(frozen_path) != FROZEN_PROBE_SHA:
        raise ValueError("Frozen source SHA mismatch")
    tree = ast.parse(frozen_path.read_text())
    worker = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "WORKER" for target in node.targets))
    worker = replace_once(worker, "import gc,json,math,os,re,signal,time,traceback,warnings",
                          "import atexit,gc,json,math,os,re,signal,time,traceback,warnings")
    worker = replace_once(worker, "SOL_H3_REMAINING_WORKER", "SOL_H3_DIAGNOSTIC_WORKER")
    worker = replace_once(worker, "start=time.monotonic();data={};cpu={};torch=None;initialized=False",
                          "assert cfg['case'] in " + repr(tuple(CASES)) + "\n"
                          "start=time.monotonic();data={};cpu={};torch=None;initialized=False\n"
                          "def exit_marker():\n"
                          " print(json.dumps({'lifecycle':'python_atexit_callback','pid':os.getpid(),"
                          "'monotonic':time.monotonic()}),flush=True)\n"
                          "atexit.register(exit_marker)")
    worker = replace_once(worker, " case=cfg['case']\n if case=='SDPA_BF16_mask128':",
                          " case=cfg['case']\n if case=='NPU_init_tiny_alloc_free':\n"
                          "  cpu.update(tiny=torch.ones(16,dtype=torch.float32));reference=cpu['tiny']\n"
                          " elif case=='SDPA_BF16_mask128':")
    begin = worker.index(" elif case=='Cholesky_triSolve_FP32_128':")
    end = worker.index(" elif case=='depthwise_Conv2d_BF16_C128_k5':", begin)
    worker = worker[:begin] + worker[end:]
    begin = worker.index(" elif case=='Cholesky_triSolve_FP32_128':")
    end = worker.index(" elif case=='depthwise_Conv2d_BF16_C128_k5':", begin)
    worker = worker[:begin] + worker[end:]
    worker = replace_once(worker, " step(phase,copy_inputs)",
                          " step(phase,copy_inputs,torch.float32 if case=='NPU_init_tiny_alloc_free' else None)")
    worker = replace_once(worker, " save();print(json.dumps({'worker_final':report}),flush=True)",
                          " report['final_report_emitted_monotonic']=time.monotonic()\n"
                          " save();print(json.dumps({'worker_final':report}),flush=True)\n"
                          " print(json.dumps({'lifecycle':'worker_final_report_emitted','pid':os.getpid(),"
                          "'monotonic':report['final_report_emitted_monotonic']}),flush=True)")
    ast.parse(worker)
    return worker


def utc_age(record):
    created = datetime.fromisoformat(record["created_utc"].replace("Z", "+00:00"))
    if created.tzinfo is None:
        raise ValueError("Preflight timestamp must include timezone")
    return (datetime.now(timezone.utc) - created).total_seconds()


def validate_preflight(path, hashes, roots):
    record = json.loads(path.read_text())
    required = {"schema": "sol_h3.primitive_cpu_preflight.v1", "status": "passed", "passed": True,
                "physical_card": PHYSICAL_CARD, "project_root": str(PROJECT),
                "torch_npu_initialized": False, "loader_preflight_passed": True,
                "legacy_import_passed": True, "runtime_roots": roots, **hashes}
    if any(record.get(key) != value for key, value in required.items()):
        raise ValueError("CPU preflight/source/runtime status mismatch")
    if not 0 <= utc_age(record) <= 3600:
        raise ValueError("CPU preflight must be fresh (at most one hour)")
    evidence = record.get("evidence", [])
    if not evidence:
        raise ValueError("CPU preflight must reference actual loader/import evidence files")
    for entry in evidence:
        source = Path(entry["path"]).resolve(strict=True)
        if not source.is_relative_to(PROJECT / "results") or sha(source) != entry["sha256"]:
            raise ValueError("CPU evidence path/hash mismatch")
    return record


def case_succeeded(entry):
    return (entry.get("operator_checks_passed") is True and entry.get("process_exit_passed") is True
            and entry.get("cleanup_passed") is True
            and entry.get("cpu_fallback_classification", {}).get("cpu_fallback_observed") is False
            and not entry.get("fallback_group_kill") and not entry.get("group_kill_errors")
            and not entry.get("group_reap_errors"))


def coordinator_succeeded(report):
    entries = report["cases"]
    names = [entry["case"] for entry in entries]
    return (len(names) == len(CASES) and len(set(names)) == len(CASES) and set(names) == set(CASES)
            and all(entry["status"] == "passed" and case_succeeded(entry)
                    and entry.get("npu_smi_after", {}).get("card_idle") is True
                    for entry in entries)
            and not report.get("coordinator_error") and not report.get("coordinator_exception_type")
            and not report.get("TERM_received") and report.get("shared_failure") is None
            and not report.get("group_kill_errors") and not report.get("group_reap_errors"))


def coordinator():
    if signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, set()):
        raise SystemExit("Inherited blocked SIGTERM rejected before any worker launch")
    cfg = json.loads(os.environ["SOL_H3_DIAGNOSTIC_COORDINATOR"])
    claim = json.loads(Path(cfg["claim"]).read_text())
    if (claim["phase"] != "claimed" or claim["claim_token"] != cfg["claim_token"]
            or claim["authorization_token_sha256"] != cfg["authorization_token_sha256"]
            or sha(Path(__file__)) != cfg["runner_sha256"]):
        raise SystemExit("New diagnostic claim/source mismatch")
    frozen = Path(cfg["frozen_probe"])
    worker = build_worker(frozen)
    if hashlib.sha256(worker.encode()).hexdigest() != cfg["worker_sha256"]:
        raise SystemExit("Generated worker SHA mismatch")
    started = time.monotonic(); deadline = started + SHARED_SECONDS
    active = {}; shared_failure = None; owned_groups = set()
    report = {"scope": SCOPE, "cases": [], "active_child_groups": [],
              "shared_seconds": SHARED_SECONDS, "physical_card": PHYSICAL_CARD, "passed": False,
              "group_cleanup_events": [], "group_kill_errors": [], "group_reap_errors": []}
    output = Path(cfg["coordinator_output"])
    def save():
        report.update(wall_seconds=time.monotonic()-started,
                      remaining_shared_seconds=max(0., deadline-time.monotonic()), active_child_groups=list(active),
                      owned_process_groups_seen=sorted(owned_groups), shared_failure=shared_failure)
        atomic_json(output, report)
    def cleanup_groups(groups, path):
        groups = list(groups); owned_groups.update(groups)
        errors = [{**error, "cleanup_path": path} for error in kill_groups(groups)]
        report["group_kill_errors"].extend(errors)
        report["group_cleanup_events"].append({"cleanup_path": path, "owned_group_ids": groups,
                                               "kill_errors": errors})
        return errors
    def terminated(signum, frame):
        report["TERM_received"] = True; cleanup_groups(active, "coordinator_TERM"); save()
        raise TimeoutError("Coordinator TERM; active groups killed")
    signal.signal(signal.SIGTERM, terminated); save()
    try:
        for index, (case, cap) in enumerate(CASES.items()):
            remaining = deadline-time.monotonic()
            if remaining < 12:
                report["cases"].append({"case": case, "status": "not_run_budget"}); save(); continue
            try: before = idle_snapshot(min(2., remaining-8.))
            except subprocess.TimeoutExpired: before = {"card_idle": False, "error": "Pre-case npu-smi timeout"}
            if not before["card_idle"]:
                shared_failure = "Card not confirmed idle before case"
                report["cases"].append({"case": case, "status": "not_run_card_not_idle", "npu_smi_before": before}); save(); break
            limit = min(float(cap), deadline-time.monotonic()-6.)
            if limit < 8:
                report["cases"].append({"case": case, "status": "not_run_budget"}); save(); continue
            env = dict(os.environ, SOL_H3_DIAGNOSTIC_WORKER=json.dumps({"case": case, "output": cfg["case_outputs"][index]}))
            command = [cfg["timeout_binary"], "--signal=TERM", "--kill-after=2s", f"{limit:.3f}s",
                       cfg["python"], "-B", "-c", worker]
            began = time.monotonic()
            old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
            try:
                process = subprocess.Popen(command, env=env, cwd=PROJECT, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE, text=True, start_new_session=True,
                                           preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask - {signal.SIGTERM}))
                active[process.pid] = process
                owned_groups.add(process.pid)
                entry = {"case": case, "status": "running", "group_id": process.pid,
                         "case_TERM_limit_seconds": limit, "requested_case_cap_seconds": cap,
                         "npu_smi_before": before}
                report["cases"].append(entry); save()
            finally: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
            try:
                stdout, stderr = process.communicate(timeout=min(limit+2.5, max(.1, deadline-time.monotonic())))
            except subprocess.TimeoutExpired:
                entry["group_kill_errors"] = cleanup_groups([process.pid], f"case_timeout:{case}")
                entry["fallback_group_kill"] = True
                try: stdout, stderr = process.communicate(timeout=.5)
                except subprocess.TimeoutExpired as error:
                    reap_error = {"group_id": process.pid, "cleanup_path": f"case_timeout:{case}",
                                  "error": str(error), "exception_type": type(error).__name__}
                    entry["group_reap_errors"] = [reap_error]; report["group_reap_errors"].append(reap_error)
                    stdout, stderr = "", "Pipes failed to close after group SIGKILL"
                    if process.stdout: process.stdout.close()
                    if process.stderr: process.stderr.close()
            observed_exit = time.monotonic()
            entry.update(exit=process.returncode, worker_wall_seconds=observed_exit-began, stdout=stdout, stderr=stderr)
            worker_output = Path(cfg["case_outputs"][index])
            if worker_output.is_file():
                try: entry["worker_report"] = json.loads(worker_output.read_text())
                except (OSError, json.JSONDecodeError) as error: entry["worker_report_error"] = str(error)
            status = entry.get("worker_report", {})
            classification = classify_cpu_fallback_logs(stdout=stdout, stderr=stderr)
            entry["cpu_fallback_classification"] = classification
            entry["operator_checks_passed"] = status.get("passed") is True
            entry["process_exit_passed"] = process.returncode == 0 and not entry.get("fallback_group_kill")
            entry["tracked_allocator_observation"] = {
                "allocated_after_cleanup": status.get("tracked_allocated_after_cleanup"),
                "reserved_after_cleanup": status.get("tracked_reserved_after_cleanup"),
                "scope": "Torch tracked allocator counters; physical device process exit is checked separately"}
            entry["cleanup_passed"] = (status.get("resource_cleanup_completed") is True and not status.get("cleanup_error")
                                       and status.get("tracked_allocated_after_cleanup") == 0
                                       and status.get("tracked_reserved_after_cleanup") == 0)
            entry["status"] = "passed" if case_succeeded(entry) else "failed"
            final_report_time = status.get("final_report_emitted_monotonic")
            if isinstance(final_report_time, (float, int)):
                entry["final_report_to_observed_pipe_close_seconds"] = max(0., observed_exit-final_report_time)
                entry["shutdown_interpretation"] = "normal_process_exit" if process.returncode == 0 else "process_failure_after_final_report"
            entry["python_atexit_observed"] = '"lifecycle": "python_atexit_callback"' in stdout
            entry["shutdown_measurement_limit"] = "Markers observe final report and Python atexit, not individual C++ destructors"
            if status.get("fatal_stage") in {"framework_import", "one_card_setup"}:
                shared_failure = {"case": case, "fatal_stage": status["fatal_stage"], "error": status.get("error")}
            if process.returncode is None:
                shared_failure = "Owned worker process exit/reap not confirmed"
            else:
                active.pop(process.pid, None)
            save()
            remaining = deadline-time.monotonic()
            if remaining <= .1:
                entry["npu_smi_after"] = {"card_idle": False, "error": "Shared deadline exhausted"}
            else:
                try: entry["npu_smi_after"] = idle_snapshot(min(2., remaining))
                except subprocess.TimeoutExpired: entry["npu_smi_after"] = {"card_idle": False, "error": "Post-case npu-smi timeout"}
            save()
            if not entry["npu_smi_after"]["card_idle"]:
                shared_failure = "Case device-process exit not confirmed"; break
            if shared_failure is not None: break
    except BaseException as error:
        report.update(coordinator_error=str(error), coordinator_exception_type=type(error).__name__)
    finally:
        cleanup_groups(active, "coordinator_finally")
        for process in list(active.values()):
            try:
                remaining = max(0., deadline-time.monotonic())
                if remaining > 0: process.wait(timeout=min(.2, remaining))
                elif process.poll() is None: raise subprocess.TimeoutExpired(process.args, 0)
            except (subprocess.TimeoutExpired, OSError) as error:
                report["group_reap_errors"].append({"group_id": process.pid, "cleanup_path": "coordinator_finally",
                    "exception_type": type(error).__name__, "error": str(error)})
        active.clear()
        represented = {entry["case"] for entry in report["cases"]}
        for case in CASES:
            if case not in represented:
                report["cases"].append({"case": case, "status": "not_run_precondition" if shared_failure is not None else "not_run_budget_or_cleanup",
                                        "shared_failure": shared_failure})
        report["shared_failure"] = shared_failure
        report["passed"] = coordinator_succeeded(report)
        save(); print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-only", action="store_true", help="Default; no claim/device/framework operations")
    parser.add_argument("--coordinator", action="store_true", help=argparse.SUPPRESS)
    for name in ("project-root", "runtime-root", "opp-root", "api-root", "environment-helper", "cpu-preflight", "authorization", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--authorization-token")
    args = parser.parse_args()
    if args.coordinator: return coordinator()
    script = Path(__file__).resolve(strict=True)
    frozen = script.with_name("probe_remaining_ops_budget_30778.py")
    classifier = script.with_name("classify_cpu_fallback_logs.py")
    worker = build_worker(frozen)
    if sha(classifier) != CLASSIFIER_SHA:
        raise SystemExit("Reviewed classifier SHA mismatch")
    hashes = {"runner_sha256": sha(script), "worker_sha256": hashlib.sha256(worker.encode()).hexdigest(),
              "classifier_sha256": CLASSIFIER_SHA, "frozen_probe_sha256": FROZEN_PROBE_SHA}
    report = {"scope": SCOPE, "check_only": not args.execute, "execution_performed": False,
              "physical_card": PHYSICAL_CARD, "server_port": 30778, "case_caps_seconds": CASES,
              "authorized_phase_limit_seconds": TOTAL_SECONDS, "shared_seconds": SHARED_SECONDS,
              "outer_TERM_seconds": OUTER_TERM, "outer_KILL_seconds": OUTER_KILL,
              "main_wait_seconds": MAIN_WAIT, "fallback_reap_seconds": MAIN_REAP,
              "final_idle_timeout_seconds": FINAL_IDLE_TIMEOUT,
              "planned_device_envelope_wait_reap_final_snapshot_seconds": MAIN_WAIT+MAIN_REAP+FINAL_IDLE_TIMEOUT,
              "phase_limit_reserve_seconds": TOTAL_SECONDS-(MAIN_WAIT+MAIN_REAP+FINAL_IDLE_TIMEOUT),
              "required_caller_outer_timeout": {"TERM_seconds": 298, "KILL_after_seconds": 2, "total_seconds": 300},
              "complete_wall_bound_requires_caller_outer_timeout": True,
              "metadata_tail_IO_excluded_from_device_envelope_measurement": True,
              "case_cap_sum_seconds": sum(CASES.values()), "case_limits_may_shorten_for_shared_budget": True,
              "old_120s_claim_or_budget_reused": False, "automatic_30min_expansion": False,
              "new_phase_allocation_record_required": True, "models_or_checkpoints_loaded": False,
              "installation_performed": False, "contract_template_is_not_authorization": True, "hashes": hashes}
    required_names = ("project_root", "runtime_root", "opp_root", "api_root", "environment_helper", "cpu_preflight", "authorization", "output")
    if not args.execute and any(getattr(args, name) is None for name in required_names):
        report["check_status"] = "source_and_budget_contract_only; remote preflight/authorization not supplied"
        report["authorization_schema"] = {"schema": "sol_h3.primitive_authorization.v1", "authorized": True,
            "authorization_token": "NEW_PARENT_SUPPLIED_TOKEN_REQUIRED", "scope": SCOPE, "project_root": str(PROJECT),
            "server_port": 30778, "physical_card": PHYSICAL_CARD, "budget_seconds": TOTAL_SECONDS,
            "max_fresh_workers": 4, "case_caps_seconds": CASES, **hashes,
            "environment_helper_sha256": "REQUIRED", "cpu_preflight_sha256": "REQUIRED"}
        report["cpu_preflight_schema"] = {"schema": "sol_h3.primitive_cpu_preflight.v1", "status": "passed",
            "passed": True, "created_utc": "FRESH_TIMEZONE_AWARE_TIMESTAMP_REQUIRED", "physical_card": PHYSICAL_CARD,
            "project_root": str(PROJECT), "torch_npu_initialized": False, "loader_preflight_passed": True,
            "legacy_import_passed": True, "runtime_roots": "EXACT_RESOLVED_ROOTS_REQUIRED", **hashes,
            "environment_helper_sha256": "REQUIRED", "evidence": [{"path": "ACTUAL_PROJECT_RESULTS_CPU_PROOF", "sha256": "REQUIRED"}]}
        print(json.dumps(report, indent=2)); return 0
    if any(getattr(args, name) is None for name in required_names):
        parser.error("Explicit project/runtime/helper/preflight/authorization/output paths required for execution")
    project = args.project_root.resolve(strict=True)
    if project != PROJECT:
        raise SystemExit("Only the approved 30778 project is in scope")
    paths = {name: getattr(args, name).resolve(strict=True) for name in required_names if name not in {"project_root", "output"}}
    if any(not value.is_relative_to(PROJECT) for value in paths.values()):
        raise SystemExit("All roots/proof/helper records must stay inside project")
    if not paths["environment_helper"].is_relative_to(PROJECT / "scripts") or any(
            not paths[name].is_relative_to(PROJECT / "results") for name in ("cpu_preflight", "authorization")):
        raise SystemExit("Helper/proof file scope mismatch")
    output = args.output.absolute()
    if not output.parent.resolve(strict=True).is_relative_to(PROJECT / "results"):
        raise SystemExit("Fresh output must stay inside project/results")
    coordinator_output = output.with_name(output.stem + ".coordinator.json")
    case_outputs = [output.with_name(output.stem + f".case{index+1}.json") for index in range(4)]
    if any(path.exists() or path.is_symlink() for path in (output, output.with_suffix(".log"), coordinator_output, *case_outputs)):
        raise SystemExit("Fresh output paths required")
    roots = {name: str(paths[name]) for name in ("runtime_root", "opp_root", "api_root")}
    hashes["environment_helper_sha256"] = sha(paths["environment_helper"])
    validate_preflight(paths["cpu_preflight"], hashes, roots)
    authorization = json.loads(paths["authorization"].read_text())
    expected = {"schema": "sol_h3.primitive_authorization.v1", "authorized": True, "scope": SCOPE,
        "project_root": str(PROJECT), "server_port": 30778, "physical_card": PHYSICAL_CARD,
        "budget_seconds": TOTAL_SECONDS, "max_fresh_workers": 4, "case_caps_seconds": CASES,
        "cpu_preflight_sha256": sha(paths["cpu_preflight"]), **hashes}
    if any(authorization.get(key) != value for key, value in expected.items()):
        raise SystemExit("New explicit authorization/source/budget/preflight mismatch")
    token = authorization.get("authorization_token", "")
    if not isinstance(token, str) or len(token) < 32 or token.startswith("NEW_PARENT"):
        raise SystemExit("A real new parent-supplied authorization token is required")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    claim = PROJECT / ("results/.primitive_diagnostics_300s." + token_hash + ".claim.json")
    report.update(authorization_path=str(paths["authorization"]), authorization_sha256=sha(paths["authorization"]),
                  cpu_preflight_path=str(paths["cpu_preflight"]), cpu_preflight_sha256=sha(paths["cpu_preflight"]),
                  claim_path=str(claim), claim_already_exists=claim.exists() or claim.is_symlink())
    if not args.execute:
        report["check_status"] = "source/preflight/authorization contract validated; no device or claim operation"
        print(json.dumps(report, indent=2)); return 0
    if args.authorization_token != token:
        raise SystemExit("Execution needs matching new authorization token argument")
    if signal.SIGTERM in signal.pthread_sigmask(signal.SIG_BLOCK, set()):
        raise SystemExit("Inherited blocked SIGTERM rejected before claim/device execution")
    spec = importlib.util.spec_from_file_location("sol_h3_diagnostic_env", paths["environment_helper"])
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    env, libraries = helper.prepare_environment(PROJECT, paths["runtime_root"], paths["opp_root"], paths["api_root"], PHYSICAL_CARD)
    timeout = shutil.which("timeout"); python = PROJECT / ".venv/bin/python"
    if not timeout or not python.is_file():
        raise SystemExit("GNU timeout and existing project Python required")
    cfg = {**hashes, "project": str(PROJECT), "frozen_probe": str(frozen), "claim": str(claim),
           "claim_token": uuid.uuid4().hex, "authorization_token_sha256": token_hash,
           "coordinator_output": str(coordinator_output), "case_outputs": [str(path) for path in case_outputs],
           "timeout_binary": timeout, "python": str(python)}
    descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    claim_value = {**cfg, "phase": "claimed", "scope": SCOPE, "new_phase_limit_seconds": TOTAL_SECONDS,
                   "authorization_sha256": report["authorization_sha256"], "created_utc": datetime.now(timezone.utc).isoformat()}
    with os.fdopen(descriptor, "w") as stream: json.dump(claim_value, stream, indent=2); stream.write("\n")
    env["SOL_H3_DIAGNOSTIC_COORDINATOR"] = json.dumps(cfg)
    command = [timeout, "--signal=TERM", "--kill-after=2s", "285s", str(python), "-B", str(script), "--coordinator"]
    began = time.monotonic()
    wait_deadline = began + MAIN_WAIT
    finish_deadline = began + MAIN_WAIT + MAIN_REAP + FINAL_IDLE_TIMEOUT
    process = None
    owned_groups = set()
    cleanup_errors = []
    stdout, stderr = "", ""
    def stop_all():
        if process is None:
            return
        owned_groups.add(process.pid)
        cleanup_errors.extend(kill_groups([process.pid]))
        groups = []
        if coordinator_output.is_file():
            try: groups.extend(json.loads(coordinator_output.read_text()).get("active_child_groups", []))
            except (OSError, json.JSONDecodeError) as error:
                report["cleanup_pid_journal_read_error"] = str(error)
        owned_groups.update(groups)
        cleanup_errors.extend(kill_groups(groups))
    def terminated(signum, frame):
        stop_all(); raise TimeoutError("Outer main TERM; all groups killed")
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM})
    previous = signal.signal(signal.SIGTERM, terminated)
    try:
        try:
            process = subprocess.Popen(command, env=env, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, start_new_session=True,
                                       preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, old_mask - {signal.SIGTERM}))
            owned_groups.add(process.pid)
            report["owned_coordinator_group"] = process.pid
            claim_value["owned_coordinator_group"] = process.pid
            atomic_json(claim, claim_value)
        except BaseException:
            stop_all()
            raise
        finally:
            # Deliver queued TERM only after the handler and owned PID exist.
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        remaining = wait_deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Coordinator spawn consumed the absolute main wait deadline")
        stdout, stderr = process.communicate(timeout=remaining)
    except BaseException as error:
        stop_all()
        report.update(outer_fallback_error=str(error), outer_exception_type=type(error).__name__)
        if process is not None:
            reap_time = min(float(MAIN_REAP), max(0., finish_deadline-time.monotonic()-FINAL_IDLE_TIMEOUT))
            try: stdout, stderr = process.communicate(timeout=reap_time)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", "Output pipes failed to close after owned group kill"
                if process.stdout: process.stdout.close()
                if process.stderr: process.stderr.close()
    finally:
        # This also covers PID-journal/report-write failures after a real spawn.
        stop_all()
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        signal.signal(signal.SIGTERM, previous)
    report.update(execution_performed=process is not None, outer_exit=process.returncode if process else None,
                  coordinator_stdout=stdout, coordinator_stderr=stderr, api_libraries=libraries,
                  owned_process_groups_seen=sorted(owned_groups), group_cleanup_errors=cleanup_errors,
                  main_spawn_seconds_in_absolute_budget=True)
    if coordinator_output.is_file():
        try: report["coordinator_report"] = json.loads(coordinator_output.read_text())
        except (OSError, json.JSONDecodeError) as error: report["coordinator_report_error"] = str(error)
    idle_time = min(float(FINAL_IDLE_TIMEOUT), max(0., finish_deadline-time.monotonic()))
    if idle_time <= 0:
        report["final_npu_smi"] = {"card_idle": False, "error": "Absolute budget exhausted before final idle check"}
    else:
        try: report["final_npu_smi"] = idle_snapshot(idle_time)
        except subprocess.TimeoutExpired: report["final_npu_smi"] = {"card_idle": False, "error": "Final npu-smi timeout"}
    report["observed_device_envelope_seconds"] = time.monotonic()-began
    report["observed_device_envelope_scope"] = "Main spawn begins through final idle snapshot; excludes report/log/claim/console tail IO"
    report["selected_card_processes_gone"] = report["final_npu_smi"]["card_idle"]
    report["passed"] = (process is not None and process.returncode == 0 and not cleanup_errors
                        and "outer_fallback_error" not in report and report.get("coordinator_report", {}).get("passed") is True
                        and report["selected_card_processes_gone"] and report["observed_device_envelope_seconds"] <= TOTAL_SECONDS)
    claim_value.update(phase="complete", report=str(output), observed_device_envelope_seconds=report["observed_device_envelope_seconds"],
                       selected_card_processes_gone=report["selected_card_processes_gone"],
                       global_ledger_requires_caller_measured_total_wall_seconds=True,
                       continuation_requires_recorded_phase_allocation=True,
                       allocation_may_draw_from_existing_approved_global_budget=True)
    try:
        atomic_json(output, report); output.with_suffix(".log").write_text(stdout+stderr)
        atomic_json(claim, claim_value)
    except OSError as error:
        stop_all()
        report.update(passed=False, record_write_error=str(error), owned_process_groups_seen=sorted(owned_groups),
                      group_cleanup_errors=cleanup_errors)
    summary = {key: report.get(key) for key in ("scope", "passed", "outer_exit", "observed_device_envelope_seconds",
                "selected_card_processes_gone", "owned_process_groups_seen", "group_cleanup_errors", "record_write_error")}
    summary["full_report_path"] = str(output)
    print(json.dumps(summary), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
