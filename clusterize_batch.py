import argparse
import concurrent.futures
import glob
import random
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch launcher for ppcx_mcmc_clustering.py -- run many reference dates."
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dates",
        help="Comma separated list of reference dates (YYYY-MM-DD). Example: 2020-01-01,2020-02-02",
    )
    group.add_argument(
        "--dates-file",
        help="File with one date (YYYY-MM-DD) per line.",
    )
    group.add_argument(
        "--random",
        type=int,
        help="Pick N random dates between --start and --end (inclusive).",
    )

    p.add_argument("--start", help="Start date for random sampling (YYYY-MM-DD).")
    p.add_argument("--end", help="End date for random sampling (YYYY-MM-DD).")
    p.add_argument(
        "--season",
        help="Optional season to restrict random sampling, format 'M-M' (months numeric 1-12 inclusive). Example: --season 6-10 for Jun-Oct.",
    )
    p.add_argument(
        "--exclude-months",
        help="Optional months to exclude from random sampling. Format: comma separated months/ranges, e.g. '1-5,11-12' to exclude Jan-May and Nov-Dec.",
    )
    p.add_argument(
        "--script-path",
        default="ppcx_mcmc_clustering.py",
        help="Path to the clustering script to run (default: ppcx_mcmc_clustering.py).",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to invoke (default: current interpreter).",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel processes to run simultaneously (default 1).",
    )
    p.add_argument(
        "--central-summary",
        default="summary_batch",
        help="Folder where summary pngs will be collected (no subfolders).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout (seconds) for each run subprocess (default none).",
    )
    return p.parse_args()


def load_dates_from_file(path: Path):
    dates = []
    with open(path) as fh:
        for r in fh:
            s = r.strip()
            if not s:
                continue
            dates.append(s)
    return dates


def random_dates_between(
    start: str, end: str, n: int, include_months=None, exclude_months=None
):
    """
    Return n unique random dates between start and end (inclusive).
    include_months / exclude_months: sets of month ints (1..12). If include_months is provided it is used;
    otherwise exclude_months (if provided) is applied.
    """
    sd = datetime.strptime(start, "%Y-%m-%d")
    ed = datetime.strptime(end, "%Y-%m-%d")
    if ed < sd:
        raise ValueError("end must be >= start")
    days = (ed - sd).days + 1
    # build candidate list of dates respecting month filters
    candidates = []
    for i in range(days):
        d = sd + timedelta(days=i)
        m = d.month
        if include_months is not None:
            if m in include_months:
                candidates.append(d)
        elif exclude_months is not None:
            if m not in exclude_months:
                candidates.append(d)
        else:
            candidates.append(d)
    if not candidates:
        raise ValueError("No candidate dates available after applying month filters.")
    if n > len(candidates):
        raise ValueError(
            f"Requested {n} dates but only {len(candidates)} candidates available."
        )
    picked = sorted({random.choice(candidates) for _ in range(n)})
    # format as YYYY-MM-DD strings
    return [d.strftime("%Y-%m-%d") for d in picked]


def _parse_months_spec(spec: str):
    """
    Parse a months spec like '1-5,11-12,7' into a set of ints {1,2,3,4,5,11,12,7}
    """
    out = set()
    if not spec:
        return out
    parts = spec.split(",")
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            if a_i <= b_i:
                out.update(range(a_i, b_i + 1))
            else:
                # wrap-around e.g. 11-2 -> 11,12,1,2
                out.update(list(range(a_i, 13)) + list(range(1, b_i + 1)))
        else:
            out.add(int(p))
    # keep only valid months
    return {m for m in out if 1 <= m <= 12}


def find_and_copy_summary(date_str: str, central_dir: Path):
    # Search recursively for summary files that contain the date string and end with _summary.png
    pattern = f"**/*{date_str}*_summary.png"
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        # also try more generic search (contain date anywhere and summary)
        hits = [
            p for p in glob.glob("**/*summary.png", recursive=True) if date_str in p
        ]
    copied = []
    for p in hits:
        src = Path(p)
        dst = central_dir / src.name
        try:
            shutil.copy(src, dst)
            copied.append(dst)
        except Exception as exc:
            print(f"[WARN] Failed to copy {src} -> {dst}: {exc}")
    return copied


def run_one_date(python: str, script: str, date: str, timeout=None):
    cmd = [python, script, "--reference-date", date]
    print(f"[RUN] {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        ok = proc.returncode == 0
        if not ok:
            print(
                f"[ERR] Date {date} exited {proc.returncode}. stdout/stderr:\n{proc.stdout}\n{proc.stderr}"
            )
        else:
            print(f"[OK] Date {date} completed.")
        return ok, proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        print(f"[ERR] Timeout for date {date}")
        return False, "timeout"


def main():
    args = parse_args()
    # build dates list
    if args.dates:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    elif args.dates_file:
        dates = load_dates_from_file(Path(args.dates_file))
    else:
        if not (args.start and args.end):
            raise SystemExit("When using --random you must provide --start and --end")
        # prepare month filters
        include_months = None
        exclude_months = None
        if args.season:
            # season provided as "M-M"
            try:
                include_months = _parse_months_spec(args.season)
            except Exception:
                raise SystemExit("Invalid --season format. Use M-M (e.g. 6-10).")
        elif args.exclude_months:
            exclude_months = _parse_months_spec(args.exclude_months)
        try:
            dates = random_dates_between(
                args.start,
                args.end,
                args.random,
                include_months=include_months,
                exclude_months=exclude_months,
            )
        except Exception as exc:
            raise SystemExit(f"Could not sample dates: {exc}")

    central_dir = Path(args.central_summary)
    central_dir.mkdir(parents=True, exist_ok=True)

    # run jobs (parallel or sequential)
    results = {}
    if args.max_workers == 1:
        for d in dates:
            ok, output = run_one_date(
                args.python, args.script_path, d, timeout=args.timeout
            )
            results[d] = ok
            # try to collect summary images for this run
            copied = find_and_copy_summary(d, central_dir)
            if copied:
                print(f"[COPY] Collected {len(copied)} summary files for {d}")
            else:
                print(f"[WARN] No summary files found for {d}")
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futs = {
                ex.submit(
                    run_one_date, args.python, args.script_path, d, args.timeout
                ): d
                for d in dates
            }
            for fut in concurrent.futures.as_completed(futs):
                d = futs[fut]
                try:
                    ok, _ = fut.result()
                except Exception as exc:
                    ok = False
                    print(f"[ERR] Exception running {d}: {exc}")
                results[d] = ok
                copied = find_and_copy_summary(d, central_dir)
                if copied:
                    print(f"[COPY] Collected {len(copied)} summary files for {d}")
                else:
                    print(f"[WARN] No summary files found for {d}")

    # summary
    succ = [d for d, ok in results.items() if ok]
    fail = [d for d, ok in results.items() if not ok]
    print(f"Done. Success: {len(succ)} failed: {len(fail)}")
    if succ:
        print("Succeeded dates:", succ)
    if fail:
        print("Failed dates:", fail)


if __name__ == "__main__":
    main()
