import argparse
import subprocess
import sys
import time
from pathlib import Path


def count_jsonl_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def fmt_seconds(s: float) -> str:
    s = int(max(0, round(s)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:d}:{sec:02d}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input JSONL (gold)")
    p.add_argument("--backend", default="ollama", help="ollama | openai")
    p.add_argument("--out_dir", default="runs", help="Directory for run outputs")
    p.add_argument("--prompt_file", required=True, help="Prompt template file path")
    p.add_argument("--host", default="http://localhost:11434")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=0, help="0 means no limit")
    p.add_argument("--print_every", type=int, default=10, help="Pass through to eval_one_model.py")
    p.add_argument("--resume", action="store_true", help="Pass through to eval_one_model.py")
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Space-separated list of model tags",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_per_run = args.limit if args.limit and args.limit > 0 else count_jsonl_lines(args.input)
    total_models = len(args.models)
    total_examples = n_per_run * total_models

    t_all = time.time()
    per_ex_times = []  # seconds per example for each completed model

    for idx, model in enumerate(args.models, start=1):
        safe_model = model.replace(":", "_").replace("/", "_")
        out_path = out_dir / f"scifact_dev200_{safe_model}_t{args.temperature}.jsonl"

        cmd = [
            sys.executable, "scripts/eval_one_model.py",
            "--input", args.input,
            "--output", str(out_path),
            "--backend", args.backend,
            "--model", model,
            "--host", args.host,
            "--temperature", str(args.temperature),
            "--prompt_file", args.prompt_file,
            "--print_every", str(args.print_every),
        ]
        if args.limit and args.limit > 0:
            cmd += ["--limit", str(args.limit)]
        if args.resume:
            cmd += ["--resume"]

        print(f"\n=== [{idx}/{total_models}] Running: {model} ===")
        print(f"Output: {out_path}")
        print("Cmd:", " ".join(cmd))

        t0 = time.time()
        r = subprocess.run(cmd)
        elapsed = time.time() - t0

        if r.returncode != 0:
            print(f"\n!!! Failed: {model} (return code {r.returncode})")
            return r.returncode

        sec_per_ex = elapsed / max(1, n_per_run)
        per_ex_times.append(sec_per_ex)

        done_models = idx
        done_examples = done_models * n_per_run
        remaining_examples = max(0, total_examples - done_examples)

        avg_sec_per_ex = sum(per_ex_times) / len(per_ex_times)
        eta_s = avg_sec_per_ex * remaining_examples

        print(
            f"=== Completed {model} in {fmt_seconds(elapsed)} "
            f"({sec_per_ex:.2f} s/ex). "
            f"Overall elapsed {fmt_seconds(time.time() - t_all)}. "
            f"ETA remaining {fmt_seconds(eta_s)} ==="
        )

    print(f"\nAll runs complete in {fmt_seconds(time.time() - t_all)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
