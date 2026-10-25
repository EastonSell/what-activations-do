"""Independent Linux process accounting for one non-adversarial stage.

The caller creates an empty output directory and passes an argv sequence. No
shell is used; the child runs with cwd=out_dir in a new session/process group.
stdout/stderr are exclusive files within out_dir, never unconsumed pipes.

This is sampled process accounting, NOT kernel containment: allocations and
files can overshoot between polls, short-lived descendants/spikes can escape
observation, and a hostile child can escape ancestry or write outside out_dir.
RSS sums resident pages (shared pages may be double-counted). Output accounting
covers observed regular-file logical sizes, including logs and this receipt,
not historical cumulative writes or files outside out_dir. Symlinks and special
files in out_dir fail closed. The caller must separately restrict stage I/O.
The watchdog's own memory is not included in child-process-tree RSS.

No acceptance/manifest is issued. A successful receipt only means the command
exited zero, no live descendants remained, and no observed operational check
failed. It is neither scientific acceptance nor proof of hard containment.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence


@dataclass(frozen=True)
class Limits:
    wall_seconds: float = 600.0
    rss_bytes: int = 1024 ** 3
    output_bytes: int = 128 * 1024 ** 2
    poll_seconds: float = 0.02
    stdout_name: str = "stdout.log"
    stderr_name: str = "stderr.log"


_MAX_WALL = 600.0
_MAX_RSS = 1024 ** 3
_MAX_OUTPUT = 128 * 1024 ** 2
_RECEIPT = "WATCHDOG.json"
_LIMITATION = (
    "Externally sampled process accounting, not kernel containment: resource "
    "use may overshoot between polls and short-lived spikes or escaped "
    "descendants may be missed. Child-tree RSS excludes the watchdog and may "
    "double-count shared pages. Output counts observed regular-file logical "
    "sizes inside out_dir plus captured logs and this receipt, not cumulative "
    "writes or files outside out_dir. Intended for non-adversarial children."
)


def _limits(value):
    if isinstance(value, Mapping):
        value = Limits(**value)
    if not isinstance(value, Limits):
        raise TypeError("limits must be Limits or a mapping")
    for name, ceiling in (("wall_seconds", _MAX_WALL), ("rss_bytes", _MAX_RSS),
                          ("output_bytes", _MAX_OUTPUT), ("poll_seconds", 1.0)):
        number = getattr(value, name)
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise TypeError(f"{name} must be a positive finite number")
        if not math.isfinite(number) or not 0 < number <= ceiling:
            raise ValueError(f"{name} must be in (0, {ceiling}]")
        if name.endswith("bytes") and not isinstance(number, int):
            raise TypeError(f"{name} must be an integer")
    names = []
    for name in (value.stdout_name, value.stderr_name):
        if not isinstance(name, str) or not name:
            raise ValueError("log names must be nonempty relative paths")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or path == Path("."):
            raise ValueError("log paths must stay within out_dir")
        if path.parts[0] == _RECEIPT:
            raise ValueError("log paths may not replace WATCHDOG.json")
        names.append(path)
    if names[0] == names[1] or names[0] in names[1].parents or names[1] in names[0].parents:
        raise ValueError("stdout and stderr must be distinct nonconflicting paths")
    return value


def _read_proc(pid):
    # comm may contain spaces and parentheses, so split after its final ')'.
    fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "state": fields[0], "ppid": int(fields[1]),
            "pgrp": int(fields[2]), "session": int(fields[3]),
            "start": int(fields[19]), "rss": max(0, int(fields[21])) * os.sysconf("SC_PAGE_SIZE")}


def _sample_tree(root_pid, tracked):
    processes = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            processes[pid] = _read_proc(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            if pid == root_pid or pid in tracked:
                raise
    selected = {pid for pid, p in processes.items()
                if pid == root_pid or p["pgrp"] == root_pid or p["session"] == root_pid
                or tracked.get(pid) == p["start"]}
    while True:
        descendants = {pid for pid, p in processes.items() if p["ppid"] in selected}
        expanded = selected | descendants
        if expanded == selected:
            break
        selected = expanded
    live = {}
    for pid in selected:
        p = processes[pid]
        tracked[pid] = p["start"]
        if p["state"] not in ("Z", "X"):
            live[pid] = p
    return sum(p["rss"] for p in live.values()), live


def _output_size(directory, logs):
    total, seen, issues = 0, set(), []
    stack = [directory]
    while stack:
        for entry in os.scandir(stack.pop()):
            try:
                info = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue  # A transient deleted file cannot be measured after deletion.
            if stat.S_ISDIR(info.st_mode):
                stack.append(entry.path)
            elif stat.S_ISREG(info.st_mode):
                total += info.st_size
                seen.add((info.st_dev, info.st_ino))
            else:
                issues.append(str(Path(entry.path).relative_to(directory)))
    # An unlinked/renamed captured log is still open: count its bytes via fstat.
    for log in logs:
        info = os.fstat(log.fileno())
        if (info.st_dev, info.st_ino) not in seen:
            total += info.st_size
            seen.add((info.st_dev, info.st_ino))
    return total, issues


def _kill_tree(proc, tracked):
    # killpg remains useful even when the direct child has already exited.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    # Also stop observed descendants that moved to another session/group.
    for pid, start in tuple(tracked.items()):
        try:
            if _read_proc(pid)["start"] == start:
                os.kill(pid, signal.SIGKILL)
        except (FileNotFoundError, ProcessLookupError):
            pass
    if proc.poll() is None:
        proc.kill()
    proc.wait(timeout=5)
    deadline = time.monotonic() + 5
    while True:
        _, live = _sample_tree(proc.pid, tracked)
        if not live:
            return
        for pid in live:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            raise RuntimeError("child descendants have not terminated after SIGKILL")
        time.sleep(0.01)


def _serialize_receipt(receipt, child_bytes, output_limit):
    # Include the receipt's own bytes. Iterate to the decimal-length fixed point.
    for _ in range(20):
        encoded = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        total = child_bytes + len(encoded)
        if total > output_limit and "output_limit" not in receipt["violations"]:
            receipt["violations"].append("output_limit")
            if receipt["status"] == "success":
                receipt["status"] = "output_limit"
            receipt["success"] = False
        if receipt["output_bytes"] == total:
            return encoded
        receipt["output_bytes"] = total
    raise RuntimeError("receipt size accounting did not converge")


def supervise(command, out_dir, limits, env=None):
    """Run once and return the exact dict written exclusively to WATCHDOG.json.

    limits is a Limits instance or a mapping of its fields. Limits may be
    tightened but never relaxed above the frozen protocol. out_dir must already
    exist, be a real directory, and be empty. env follows subprocess.Popen's
    semantics: None inherits; a mapping replaces the environment. The caller
    supplies any BLAS/CPU environment policy; the watchdog records no secrets.

    Invalid inputs and reused/nonfresh directories raise before launch.
    Launch, child, budget and accounting failures return a failure receipt.
    Receipt collisions or inability to confirm terminal child cleanup raise,
    never returning success. The function never overwrites any output.
    """
    limits = _limits(limits)
    if not sys.platform.startswith("linux") or not Path("/proc/self/stat").is_file():
        raise RuntimeError("Linux with readable /proc is required")
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence) or not command:
        raise TypeError("command must be a nonempty argv sequence, not shell text")
    command = [os.fspath(item) for item in command]
    if any(not isinstance(item, str) or "\x00" in item for item in command):
        raise ValueError("command arguments must be NUL-free strings")
    directory = Path(out_dir)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("out_dir must be an existing real directory")
    directory = directory.resolve()
    if any(directory.iterdir()):
        raise FileExistsError("out_dir must be empty; no retry, resume or overwrite")

    started = time.monotonic()
    proc, logs, tracked = None, [], {}
    peak_rss, peak_output, child_bytes = 0, 0, 0
    status, violations, errors = None, [], []

    def fail(reason, detail=None):
        nonlocal status
        if status is None or status == "success":
            status = reason
        if reason not in violations:
            violations.append(reason)
        if detail is not None:
            errors.append(str(detail))

    try:
        for name in (limits.stdout_name, limits.stderr_name):
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            logs.append(path.open("xb", buffering=0))
        try:
            proc = subprocess.Popen(command, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                    stdout=logs[0], stderr=logs[1], start_new_session=True)
        except OSError as exc:
            fail("launch_failure", f"{type(exc).__name__}: {exc}")
        if proc is not None:
            while True:
                rss, live = _sample_tree(proc.pid, tracked)
                peak_rss = max(peak_rss, rss)
                child_bytes, issues = _output_size(directory, logs)
                peak_output = max(peak_output, child_bytes)
                elapsed = time.monotonic() - started
                if issues:
                    fail("watchdog_error", "unsupported output entries: " + ", ".join(issues))
                if elapsed > limits.wall_seconds:
                    fail("wall_limit")
                if rss > limits.rss_bytes:
                    fail("rss_limit")
                if child_bytes > limits.output_bytes:
                    fail("output_limit")
                code = proc.poll()
                if code is not None:
                    if code != 0:
                        fail("exit_failure")
                    if any(pid != proc.pid for pid in live):
                        fail("watchdog_error", "direct child exited with live descendants")
                    if status is None:
                        status = "success"
                    break
                if status is not None:
                    break
                remaining = max(0, limits.wall_seconds - elapsed)
                time.sleep(min(limits.poll_seconds, remaining))
    except Exception as exc:
        fail("watchdog_error", f"{type(exc).__name__}: {exc}")
    except BaseException:
        # Interruption must not leave a child running. Do not pretend a receipt
        # was produced if the supervising process itself is interrupted.
        if proc is not None:
            _kill_tree(proc, tracked)
        raise
    finally:
        try:
            if proc is not None:
                _kill_tree(proc, tracked)
            try:
                child_bytes, issues = _output_size(directory, logs)
                peak_output = max(peak_output, child_bytes)
                if issues:
                    fail("watchdog_error", "unsupported terminal output entries: " + ", ".join(issues))
                if child_bytes > limits.output_bytes:
                    fail("output_limit")
            except Exception as exc:
                fail("watchdog_error", f"terminal accounting: {type(exc).__name__}: {exc}")
        finally:
            for log in logs:
                log.close()

    elapsed = time.monotonic() - started
    if elapsed > limits.wall_seconds and proc is not None:
        fail("wall_limit")
    receipt = {
        "schema_version": 1, "status": status or "watchdog_error",
        "success": status == "success", "violations": violations,
        "returncode": proc.returncode if proc is not None else None,
        "elapsed_seconds": elapsed, "peak_sampled_rss_bytes": peak_rss,
        "peak_sampled_output_bytes": peak_output, "output_bytes": child_bytes,
        "child_output_bytes": child_bytes, "command": command,
        "pid": proc.pid if proc is not None else None, "limits": asdict(limits),
        "errors": errors, "accounting_limitation": _LIMITATION,
    }
    encoded = _serialize_receipt(receipt, child_bytes, limits.output_bytes)
    # 'xb' is intentional: a child-created or concurrent receipt is a hard error.
    with (directory / _RECEIPT).open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return receipt
