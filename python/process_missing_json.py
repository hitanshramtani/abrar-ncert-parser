from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PAIR_SEPARATOR = " -> "
CHAPTER_PATTERN = re.compile(r"^chapter_(\d+)_(.+)$")


@dataclass(frozen=True)
class MissingEntry:
    pdf_path: Path
    json_path: Path
    class_num: int
    subject: str
    chapter_num: int
    chapter_name: str


@dataclass(frozen=True)
class InvalidEntry:
    pdf_path: Path
    json_path: Path
    reason: str


def _load_missing_pairs(missing_file: Path) -> list[tuple[Path, Path]]:
    if not missing_file.exists():
        raise FileNotFoundError(f"Missing file not found: {missing_file}")

    pairs: list[tuple[Path, Path]] = []
    for raw_line in missing_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if PAIR_SEPARATOR not in line:
            continue
        left, right = line.split(PAIR_SEPARATOR, 1)
        if not left.lower().endswith(".pdf"):
            continue
        pairs.append((Path(left.strip()), Path(right.strip())))
    return pairs


def _extract_metadata_from_json_path(json_path: Path, output_root: Path) -> tuple[int, str, int, str]:
    relative = json_path.relative_to(output_root)
    if len(relative.parts) < 3:
        raise ValueError(f"Invalid output path format: {json_path}")

    class_part = relative.parts[0]
    subject = relative.parts[1]
    stem = relative.stem

    class_match = re.match(r"^class_(\d+)$", class_part)
    if not class_match:
        raise ValueError(f"Invalid class folder in path: {json_path}")
    class_num = int(class_match.group(1))

    chapter_match = CHAPTER_PATTERN.match(stem)
    if not chapter_match:
        raise ValueError(f"Invalid chapter filename format: {json_path.name}")

    chapter_num = int(chapter_match.group(1))
    chapter_name = chapter_match.group(2).replace("_", " ").strip()
    if not chapter_name:
        chapter_name = "unknown"

    return class_num, subject, chapter_num, chapter_name


def _build_entries(
    missing_pairs: list[tuple[Path, Path]],
    output_root: Path,
) -> tuple[list[MissingEntry], list[InvalidEntry]]:
    entries: list[MissingEntry] = []
    invalid_entries: list[InvalidEntry] = []
    for pdf_path, json_path in missing_pairs:
        try:
            class_num, subject, chapter_num, chapter_name = _extract_metadata_from_json_path(
                json_path=json_path,
                output_root=output_root,
            )
            entries.append(
                MissingEntry(
                    pdf_path=pdf_path,
                    json_path=json_path,
                    class_num=class_num,
                    subject=subject,
                    chapter_num=chapter_num,
                    chapter_name=chapter_name,
                )
            )
        except Exception as exc:
            invalid_entries.append(
                InvalidEntry(
                    pdf_path=pdf_path,
                    json_path=json_path,
                    reason=str(exc),
                )
            )
    return entries, invalid_entries


def _run_one(main_py: Path, entry: MissingEntry) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(main_py),
        str(entry.pdf_path),
        str(entry.class_num),
        entry.subject,
        str(entry.chapter_num),
        entry.chapter_name,
        str(entry.json_path),
    ]
    return subprocess.run(cmd, text=True, capture_output=True)


def process_missing_json(
    missing_file: Path,
    output_root: Path,
    main_py: Path,
    skip_existing: bool,
    limit: int | None,
    delay_seconds: float,
    dry_run: bool,
    failed_log: Path,
) -> int:
    missing_pairs = _load_missing_pairs(missing_file)
    entries, invalid_entries = _build_entries(missing_pairs, output_root=output_root)

    if limit is not None:
        entries = entries[:limit]

    total = len(entries)
    success = 0
    skipped = 0
    failed = 0

    print(f"Found missing entries: {len(missing_pairs)}")
    if invalid_entries:
        print(f"Invalid entries skipped before run: {len(invalid_entries)}")
    if limit is not None:
        print(f"Processing limit applied: {limit}")
    print(f"Run set size: {total}")
    print("")

    failed_lines: list[str] = []
    for invalid in invalid_entries:
        failed_lines.append(
            f"INVALID | {invalid.pdf_path} -> {invalid.json_path} | reason={invalid.reason}"
        )

    for index, entry in enumerate(entries, start=1):
        if skip_existing and entry.json_path.exists():
            skipped += 1
            print(f"[{index}/{total}] SKIP already exists: {entry.json_path}")
            continue

        print(
            f"[{index}/{total}] RUN class={entry.class_num} subject={entry.subject} "
            f"chapter={entry.chapter_num} json={entry.json_path.name}"
        )

        if dry_run:
            continue

        try:
            result = _run_one(main_py=main_py, entry=entry)
        except Exception as exc:
            failed += 1
            error_text = str(exc).strip() or "unknown subprocess error"
            print(f"[{index}/{total}] FAIL exception={error_text}")
            failed_lines.append(
                f"FAILED | {entry.pdf_path} -> {entry.json_path} | reason={error_text}"
            )
            if delay_seconds > 0:
                time.sleep(delay_seconds)
            continue

        if result.returncode == 0:
            success += 1
            print(f"[{index}/{total}] OK")
        else:
            failed += 1
            print(f"[{index}/{total}] FAIL return_code={result.returncode}")
            reason = ""
            if result.stderr.strip():
                reason = result.stderr.strip()
                print(reason)
            elif result.stdout.strip():
                reason = result.stdout.strip()
                print(reason)
            else:
                reason = f"returned non-zero exit code {result.returncode}"
            failed_lines.append(
                f"FAILED | {entry.pdf_path} -> {entry.json_path} | reason={reason}"
            )

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    if failed_lines and not dry_run:
        failed_log.parent.mkdir(parents=True, exist_ok=True)
        failed_log.write_text("\n".join(failed_lines) + "\n", encoding="utf-8")

    print("")
    print("Summary:")
    print(f"Total considered: {total}")
    print(f"Success: {success}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    if invalid_entries:
        print(f"Invalid metadata entries: {len(invalid_entries)}")
    if failed_lines and not dry_run:
        print(f"Failed details written to: {failed_log}")

    return 1 if (failed > 0 or len(invalid_entries) > 0) else 0


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    explicit_default_missing = Path(
        r"C:\Users\hitan\Desktop\coding\clg_work_folder\abrar\python\missing.txt"
    )

    parser = argparse.ArgumentParser(
        description="Process missing NCERT JSON files listed in missing.txt"
    )
    parser.add_argument(
        "--missing-file",
        type=Path,
        default=explicit_default_missing,
        help="Path to missing.txt generated by check_file_exist.py",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=base_dir / "output",
        help="Output root folder containing class_x/subject/chapter_y.json",
    )
    parser.add_argument(
        "--main-py",
        type=Path,
        default=base_dir / "main.py",
        help="Path to main.py parser entry point",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Process even if the target JSON already exists",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N missing entries",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Delay between runs to reduce API pressure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would run, without executing main.py",
    )
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=base_dir / "failed_missing_runs.txt",
        help="Path to write failed/invalid entry details",
    )
    args = parser.parse_args()

    resolved_missing = args.missing_file.resolve()
    resolved_output = args.output_root.resolve()
    resolved_main = args.main_py.resolve()
    resolved_failed_log = args.failed_log.resolve()

    print(f"Using missing file: {resolved_missing}")
    print(f"Using output root: {resolved_output}")
    print(f"Using main.py: {resolved_main}")
    print(f"Failed log path: {resolved_failed_log}")
    print("")

    exit_code = process_missing_json(
        missing_file=resolved_missing,
        output_root=resolved_output,
        main_py=resolved_main,
        skip_existing=not args.no_skip_existing,
        limit=args.limit,
        delay_seconds=max(args.delay_seconds, 0.0),
        dry_run=args.dry_run,
        failed_log=resolved_failed_log,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
