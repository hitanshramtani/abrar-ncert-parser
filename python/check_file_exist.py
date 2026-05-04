from __future__ import annotations

from pathlib import Path


def check_file_exist() -> None:
    base_dir = Path(__file__).resolve().parent
    ncert_root = base_dir / "ncert_pdfs"
    output_root = base_dir / "output"
    missing_report_path = base_dir / "missing.txt"

    if not ncert_root.exists() or not output_root.exists():
        raise FileNotFoundError(
            f"Expected folders not found.\n"
            f"ncert_pdfs: {ncert_root}\n"
            f"output: {output_root}"
        )

    pdf_files = sorted(ncert_root.rglob("*.pdf"))
    json_files = sorted(output_root.rglob("*.json"))

    missing_entries: list[tuple[Path, Path]] = []

    for pdf_path in pdf_files:
        relative_pdf = pdf_path.relative_to(ncert_root)
        expected_json = (output_root / relative_pdf).with_suffix(".json")
        if not expected_json.exists():
            missing_entries.append((pdf_path, expected_json))

    lines: list[str] = [
        f"ncert_pdfs: {ncert_root}",
        f"output: {output_root}",
        f"total_pdf_files: {len(pdf_files)}",
        f"total_json_files: {len(json_files)}",
        f"missing_json_files: {len(missing_entries)}",
        "",
        "Missing entries (PDF -> expected JSON):",
    ]

    for pdf_path, expected_json in missing_entries:
        lines.append(f"{pdf_path} -> {expected_json}")

    missing_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Total PDF files: {len(pdf_files)}")
    print(f"Total JSON files: {len(json_files)}")
    print(f"Missing JSON files: {len(missing_entries)}")
    print(f"Missing report written to: {missing_report_path}")


if __name__ == "__main__":
    check_file_exist()
