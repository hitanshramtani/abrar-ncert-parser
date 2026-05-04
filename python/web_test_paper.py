from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, render_template_string, request


BASE_DIR = Path(__file__).resolve().parent
PDF_ROOT = BASE_DIR / "ncert_pdfs"
JSON_ROOT = BASE_DIR / "output"
CHAPTER_PATTERN = re.compile(r"^chapter_(\d+)_(.+)$")

app = Flask(__name__)


@dataclass(frozen=True)
class ChapterEntry:
    class_name: str
    subject: str
    chapter_file_stem: str
    chapter_number: int
    chapter_title: str


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _build_catalog() -> list[ChapterEntry]:
    entries: list[ChapterEntry] = []
    if not PDF_ROOT.exists():
        return entries

    for pdf_path in PDF_ROOT.rglob("chapter_*.pdf"):
        relative = pdf_path.relative_to(PDF_ROOT)
        if len(relative.parts) < 3:
            continue

        class_name = relative.parts[0]
        subject = relative.parts[1]
        stem = pdf_path.stem

        chapter_match = CHAPTER_PATTERN.match(stem)
        if not chapter_match:
            continue

        chapter_number = int(chapter_match.group(1))
        chapter_title = chapter_match.group(2).replace("_", " ").strip()
        entries.append(
            ChapterEntry(
                class_name=class_name,
                subject=subject,
                chapter_file_stem=stem,
                chapter_number=chapter_number,
                chapter_title=chapter_title,
            )
        )

    entries.sort(
        key=lambda item: (
            int(item.class_name.split("_")[-1]),
            item.subject.lower(),
            item.chapter_number,
            item.chapter_title.lower(),
        )
    )
    return entries


def _catalog_payload(catalog: list[ChapterEntry]) -> list[dict]:
    return [
        {
            "class_name": entry.class_name,
            "subject": entry.subject,
            "chapter_file_stem": entry.chapter_file_stem,
            "chapter_number": entry.chapter_number,
            "chapter_title": entry.chapter_title,
        }
        for entry in catalog
    ]


def _extract_questions(raw: object) -> list[dict]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        questions = raw.get("questions")
        if isinstance(questions, list):
            return [item for item in questions if isinstance(item, dict)]
    return []


def _format_question(question: dict) -> str:
    question_text = str(question.get("question_text") or "").strip()
    parts = question.get("parts") if isinstance(question.get("parts"), list) else []

    if parts:
        part_lines = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            label = str(part.get("part_label") or "").strip()
            part_text = str(part.get("part_text") or "").strip()
            if label and part_text:
                part_lines.append(f"{label} {part_text}")
            elif part_text:
                part_lines.append(part_text)
        if part_lines:
            block = "\n".join(part_lines)
            return f"{question_text}\n{block}" if question_text else block

    return question_text or "Question text not available"


def _format_answer(question: dict) -> str:
    answer_text = str(question.get("answer_text") or "").strip()
    final_answer = str(question.get("final_answer") or "").strip()
    parts = question.get("parts") if isinstance(question.get("parts"), list) else []

    part_answers: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        label = str(part.get("part_label") or "").strip()
        part_answer = str(part.get("answer_text") or part.get("final_answer") or "").strip()
        if not part_answer:
            continue
        if label:
            part_answers.append(f"{label} {part_answer}")
        else:
            part_answers.append(part_answer)

    combined: list[str] = []
    if answer_text:
        combined.append(answer_text)
    if part_answers:
        combined.extend(part_answers)
    if final_answer and final_answer not in combined:
        combined.append(final_answer)

    if not combined:
        return "Answer not available"
    return "\n".join(combined)


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Test Paper Generator</title>
  <style>
    body {
      font-family: "Segoe UI", Tahoma, sans-serif;
      max-width: 920px;
      margin: 24px auto;
      padding: 0 16px;
      background: #f4f6f8;
      color: #111827;
    }
    .card {
      background: #ffffff;
      padding: 16px;
      border-radius: 12px;
      box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
      margin-bottom: 16px;
    }
    h1 { margin-top: 0; }
    label {
      display: block;
      margin: 10px 0 4px;
      font-weight: 600;
    }
    select, input, button {
      width: 100%;
      box-sizing: border-box;
      padding: 10px;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      font-size: 14px;
    }
    button {
      margin-top: 14px;
      background: #0f766e;
      color: white;
      border: none;
      cursor: pointer;
    }
    .check-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      font-weight: 600;
    }
    .check-row input[type="checkbox"] {
      width: auto;
      padding: 0;
      margin: 0;
      transform: scale(1.1);
    }
    .missing {
      background: #fee2e2;
      color: #991b1b;
      border: 1px solid #fecaca;
      padding: 10px;
      border-radius: 8px;
      font-weight: 700;
    }
    pre {
      white-space: pre-wrap;
      background: #f9fafb;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 10px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Test Paper Generator</h1>
    <form method="post">
      <label for="class_name">Class</label>
      <select name="class_name" id="class_name"></select>

      <label for="subject">Subject</label>
      <select name="subject" id="subject"></select>

      <label for="chapter_file_stem">Chapter</label>
      <select name="chapter_file_stem" id="chapter_file_stem"></select>

      <label for="question_count">Questions Count</label>
      <input type="number" min="1" name="question_count" id="question_count" value="{{ selected_count }}">

      <label class="check-row" for="use_maximum_questions">
        <input type="checkbox" name="use_maximum_questions" id="use_maximum_questions" {% if use_maximum_questions %}checked{% endif %}>
        Use maximum questions
      </label>

      <button type="submit">Generate Test Paper</button>
    </form>
  </div>

  {% if missing %}
  <div class="missing">miising</div>
  {% endif %}

  {% if generated_questions %}
  <div class="card">
    <h2>{{ meta.class_label }} | {{ meta.subject_label }} | Chapter {{ meta.chapter_number }}: {{ meta.chapter_title }}</h2>
    <h3>Total Questions: {{ generated_questions|length }}</h3>
    {% for q in generated_questions %}
      <pre>Q{{ loop.index }}. {{ q.question }}</pre>
      <pre>Answer {{ loop.index }}. {{ q.answer }}</pre>
    {% endfor %}
  </div>
  {% endif %}

  <script>
    const catalog = {{ catalog_json | safe }};
    const selected = {
      class_name: "{{ selected_class }}",
      subject: "{{ selected_subject }}",
      chapter_file_stem: "{{ selected_chapter }}"
    };

    const classSelect = document.getElementById("class_name");
    const subjectSelect = document.getElementById("subject");
    const chapterSelect = document.getElementById("chapter_file_stem");

    function unique(items) {
      return [...new Set(items)];
    }

    function humanize(text) {
      return text.replaceAll("_", " ").replace(/\\b\\w/g, c => c.toUpperCase());
    }

    function fillClasses() {
      const classes = unique(catalog.map(item => item.class_name));
      classSelect.innerHTML = "";
      for (const className of classes) {
        const opt = document.createElement("option");
        opt.value = className;
        opt.textContent = humanize(className);
        classSelect.appendChild(opt);
      }
      if (selected.class_name && classes.includes(selected.class_name)) {
        classSelect.value = selected.class_name;
      }
    }

    function fillSubjects() {
      const className = classSelect.value;
      const subjects = unique(
        catalog
          .filter(item => item.class_name === className)
          .map(item => item.subject)
      );
      subjectSelect.innerHTML = "";
      for (const subject of subjects) {
        const opt = document.createElement("option");
        opt.value = subject;
        opt.textContent = humanize(subject);
        subjectSelect.appendChild(opt);
      }
      if (selected.subject && subjects.includes(selected.subject)) {
        subjectSelect.value = selected.subject;
      }
    }

    function fillChapters() {
      const className = classSelect.value;
      const subject = subjectSelect.value;
      const chapters = catalog
        .filter(item => item.class_name === className && item.subject === subject)
        .sort((a, b) => a.chapter_number - b.chapter_number);
      chapterSelect.innerHTML = "";
      for (const chapter of chapters) {
        const opt = document.createElement("option");
        opt.value = chapter.chapter_file_stem;
        opt.textContent = `Chapter ${chapter.chapter_number}: ${humanize(chapter.chapter_title)}`;
        chapterSelect.appendChild(opt);
      }
      if (selected.chapter_file_stem && chapters.some(c => c.chapter_file_stem === selected.chapter_file_stem)) {
        chapterSelect.value = selected.chapter_file_stem;
      }
    }

    classSelect.addEventListener("change", () => {
      fillSubjects();
      fillChapters();
    });

    subjectSelect.addEventListener("change", () => {
      fillChapters();
    });

    fillClasses();
    fillSubjects();
    fillChapters();
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    catalog = _build_catalog()
    catalog_json = json.dumps(_catalog_payload(catalog))
    entry_map = {
        (entry.class_name, entry.subject, entry.chapter_file_stem): entry for entry in catalog
    }

    default_entry = catalog[0] if catalog else None
    selected_class = request.form.get("class_name") or (default_entry.class_name if default_entry else "")
    selected_subject = request.form.get("subject") or (default_entry.subject if default_entry else "")
    selected_chapter = request.form.get("chapter_file_stem") or (
        default_entry.chapter_file_stem if default_entry else ""
    )

    try:
        selected_count = max(1, int(request.form.get("question_count", "10")))
    except ValueError:
        selected_count = 10

    use_maximum_questions = request.form.get("use_maximum_questions") == "on"
    generated_questions: list[dict[str, str]] = []
    missing = False
    meta = {
        "class_label": _humanize(selected_class),
        "subject_label": _humanize(selected_subject),
        "chapter_number": "",
        "chapter_title": "",
    }

    if request.method == "POST":
        entry = entry_map.get((selected_class, selected_subject, selected_chapter))
        if not entry:
            missing = True
        else:
            meta["chapter_number"] = entry.chapter_number
            meta["chapter_title"] = _humanize(entry.chapter_title)
            json_path = JSON_ROOT / entry.class_name / entry.subject / f"{entry.chapter_file_stem}.json"

            if not json_path.exists():
                missing = True
            else:
                try:
                    raw = json.loads(json_path.read_text(encoding="utf-8"))
                    questions = _extract_questions(raw)
                    formatted = [
                        {
                            "question": _format_question(question),
                            "answer": _format_answer(question),
                        }
                        for question in questions
                    ]
                    if formatted:
                        if use_maximum_questions:
                            generated_questions = formatted
                        else:
                            take_count = min(selected_count, len(formatted))
                            generated_questions = random.sample(formatted, take_count)
                except (OSError, json.JSONDecodeError):
                    missing = True

    return render_template_string(
        TEMPLATE,
        catalog_json=catalog_json,
        selected_class=selected_class,
        selected_subject=selected_subject,
        selected_chapter=selected_chapter,
        selected_count=selected_count,
        use_maximum_questions=use_maximum_questions,
        generated_questions=generated_questions,
        missing=missing,
        meta=meta,
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
