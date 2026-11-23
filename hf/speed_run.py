"""Run one hf.py JSON config and record wall time and memory per SCF stage.

A sampler thread reads VmRSS once a second. Every progress line that hf.py
emits is time-stamped, so the summary splits the run into the coarse-grid
stage, the fine-grid stage and the final rebuild, each with its wall time
and sampled RSS peak. VmHWM is the whole-process high-water mark.
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path


def _status_gib(field):
    try:
        text = Path("/proc/self/status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(field + ":"):
            return float(line.split()[1]) / (1024 * 1024)
    return None


def stage_summary(events, samples, end):
    """Wall time and sampled RSS peak between stage boundary messages."""

    marks = [("coarse", 0.0)]
    for seconds, message in events:
        if message.startswith("SCF switching to final grid"):
            marks.append(("fine", seconds))
        elif message == "SCF final energy":
            marks.append(("final_rebuild", seconds))
    marks.append(("end", end))
    stages = {}
    for (name, start), (_, stop) in zip(marks[:-1], marks[1:]):
        values = [rss for t, rss in samples if start <= t <= stop and rss is not None]
        stages[name] = {
            "seconds": stop - start,
            "max_sampled_rss_gib": max(values) if values else None,
        }
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hf

    origin = time.perf_counter()
    samples = []
    events = []
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            samples.append((time.perf_counter() - origin, _status_gib("VmRSS")))
            stop.wait(args.interval)

    original_emit = hf._emit_progress

    def emit(message, *rest, **kwargs):
        events.append((time.perf_counter() - origin, str(message)))
        return original_emit(message, *rest, **kwargs)

    hf._emit_progress = emit
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    summary = {"config": str(Path(args.config).resolve()), "status": "failed",
               "hf_path": str(Path(hf.__file__).resolve())}
    try:
        summary["result"] = hf.run_from_config(args.config)
        summary["status"] = "success"
    except BaseException as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        sampler.join()
        end = time.perf_counter() - origin
        summary["elapsed_seconds"] = end
        summary["vmhwm_gib"] = _status_gib("VmHWM")
        summary["stages"] = stage_summary(events, samples, end)
        summary["rss_samples"] = [[round(t, 1), round(rss, 3)]
                                  for t, rss in samples if rss is not None]
        summary["events"] = [[round(t, 3), message[:200]] for t, message in events]
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=1, default=float))
    print(f"speed_run summary: {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
