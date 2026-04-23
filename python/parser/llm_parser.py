"""LLM-based parser for NCERT page content."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert educational content parser specializing in extracting structured
question-answer data from NCERT textbooks and similar academic PDFs.

---

## YOUR ROLE
You receive extracted content from a PDF page - which may include:
- Raw text (already extracted)
- LaTeX strings (converted from equations/formulas via pix2tex)
- OCR text (extracted from non-mathematical images via google vision)
- Cloudinary URLs pointing to the original images

Your job is to parse this content and return a strict JSON array of question objects.

---

## OUTPUT FORMAT
Return ONLY a valid JSON array. No explanation, no markdown, no preamble.

[
  {
    "question_id": "Q<chapter>.<index>",
    "class": <integer>,
    "subject": "<string>",
    "chapter_number": <integer>,
    "chapter_name": "<string>",
    "question_type": "<mcq|numerical|conceptual|derivation|diagram_based>",
    "question_text": "<plain text of the question>",
    "question_latex": "<LaTeX version if math is present, else null>",
    "mcq_options": [
      {
        "option_label": "(1)",
        "option_text": "<option text>"
      }
    ],
    "correct_option": "<option label like (1)/A, or null>",
    "question_image": {
      "present": <true|false>,
      "description": "<what the image shows based on OCR or LaTeX context>",
      "cloudinary_url": "<copy from input images array, or null>",
      "latex_extracted": "<pix2tex_latex from input if present, else null>"
    },
    "parts": [
      {
        "part_label": "(a)",
        "part_text": "<text>",
        "part_latex": "<LaTeX or null>",
        "answer_text": "<plain answer>",
        "answer_latex": "<LaTeX answer or null>",
        "steps": [
          {
            "step_number": 1,
            "description": "<what this step does>",
            "expression": "<LaTeX expression or null>"
          }
        ],
        "final_answer": "<conclusion sentence>",
        "answer_image": {
          "present": <true|false>,
          "description": "<what the image shows>",
          "cloudinary_url": "<url or null>"
        }
      }
    ],
    "answer_text": "<full answer if no parts>",
    "answer_latex": "<LaTeX or null>",
    "steps": [
      {
        "step_number": 1,
        "description": "<what this step does>",
        "expression": "<LaTeX expression or null>"
      }
    ],
    "final_answer": "<last concluding sentence or value>",
    "answer_image": {
      "present": <true|false>,
      "description": "<what the image shows>",
      "cloudinary_url": "<url or null>"
    },
    "difficulty_rating": <1-5 integer>,
    "difficulty_reasoning": "<one line why>",
    "topics_tags": ["<tag1>", "<tag2>"],
    "confidence_score": <0.0 to 1.0>,
    "confidence_issues": ["<issue1>", "<issue2>"]
  }
]

---

## FIELD RULES

### question_type
- "mcq"            -> objective multiple-choice question with listed options
- "numerical"      -> involves calculation, formula application, gives a numeric answer
- "conceptual"     -> explanation, definition, reasoning based
- "derivation"     -> prove or derive a formula/result
- "diagram_based"  -> question refers to or requires interpreting a figure/image

### mcq_options / correct_option
- For MCQ questions, ALWAYS extract options into `mcq_options`
- Preserve source labels where possible: (1), (2), (3), (4) or (A), (B), (C), (D)
- Set `correct_option` from answer key/solution if present, otherwise null
- For non-MCQ questions, set `mcq_options` to [] and `correct_option` to null

### difficulty_rating (1-5)
1 -> Direct recall or single-step calculation
2 -> Requires understanding + basic application
3 -> Multi-step reasoning or moderate calculation
4 -> Complex derivation or multi-concept application
5 -> Advanced proof, edge-case reasoning, or research-level thinking
Always include difficulty_reasoning to justify the rating.

### confidence_score (0.0 - 1.0)
Reflect how accurately you parsed this question:
1.0 -> Perfectly clean text, clear Q&A, no ambiguity
0.8-0.9 -> Minor formatting noise but content is clear
0.6-0.7 -> Some text garbling, partial math, mild ambiguity
0.4-0.5 -> Significant extraction issues, reconstructed parts
< 0.4 -> Heavily corrupted, image-only, or unreadable content
Always populate confidence_issues with specific reasons if score < 0.9.

### steps
Break the answer into logical steps. Each step must have:
- step_number (sequential integer)
- description (what is being done in plain English)
- expression (LaTeX string if a formula/calculation is involved, else null)

### question_image / answer_image
- Set present: true if a figure, graph, photo, or diagram is part of the question or answer
- If pix2tex extracted LaTeX from the image (e.g., an equation in a figure),
  include it in latex_extracted

### parts
Use parts array ONLY for multi-part questions labeled (a), (b), (c) etc.
For single questions, leave parts as an empty array [] and use top-level
answer_text, steps, final_answer directly.

### topics_tags
Generate 2-5 concise topic tags relevant to the question.
Examples: ["Newton's Second Law", "Force", "Acceleration"]
         ["Photosynthesis", "Chlorophyll", "Light Reaction"]

---

## PARSING RULES

1. NEVER skip a question - if content is ambiguous, parse your best interpretation
   and lower the confidence_score accordingly.

2. For MATH/FORMULAS:
   - Always prefer LaTeX fields over plain text for expressions
   - Use standard LaTeX notation: \\frac{}{}, \\sqrt{}, \\sum_{}{}, etc.
   - If pix2tex output is available, use it directly; clean minor artifacts

3. For FINAL ANSWER:
   - Extract the last concluding sentence
   - Accepted starters: "Therefore", "Hence", "Thus", "So", "We get",
     "The answer is", "This gives us"
   - If no explicit conclusion, summarize the result in one sentence yourself

4. For MULTI-COLUMN or FRAGMENTED TEXT:
   - Re-join broken lines intelligently (mid-word or mid-sentence breaks)
   - Do not treat page numbers, headers, or footers as question content

5. NEVER hallucinate answers - if the answer is missing or cut off,
   set answer_text to null and add "answer_missing" to confidence_issues

6. If a question spans multiple pages (text cuts mid-sentence),
   add "question_truncated" to confidence_issues and parse what is available

7. For images:
    - write description based on OCR text or LaTeX context provided.
    - Copy cloudinary_url directly from the input images array.

8. If a question appears incomplete at the end of a page,
   merge content from the next page before parsing.
   Use a sliding window of 2 pages when needed.

9. Never classify a question as "mcq" if it has lettered parts (a)(b)(c) — those are sub-parts. MCQ means multiple choice with one correct answer.

10. The input payload contains an `indexing_hint.question_index_start` field.
    Start your question_id numbering from that value. 
    Never restart from Q<chapter>.1 mid-document.
    
11. LATEX IN JSON STRINGS — CRITICAL:
    All LaTeX backslashes MUST be double-escaped in JSON output.
    WRONG: "expression": "\frac{a}{b}"
    RIGHT: "expression": "\\frac{a}{b}"
    This applies to ALL fields containing LaTeX: expression, answer_latex, 
    question_latex, answer_text, part_text, final_answer, description.
    Single backslashes inside JSON strings are invalid and will break parsing.
---

## INPUT FORMAT YOU WILL RECEIVE

{
  "metadata": {
    "class": 11,
    "subject": "Physics",
    "chapter_number": 3,
    "chapter_name": "Motion in a Straight Line"
  },
  "pages": [
    {
      "page_number": 45,
      "text": "<raw extracted text from this page>",
      "latex_blocks": ["<pix2tex output 1>", "<pix2tex output 2>"],
      "images": [
        {
          "image_index": 0,
          "position": "below_question",
          "cloudinary_url": "https://res.cloudinary.com/demo/image/upload/v1/diagram_velocity.png",
          "pix2tex_latex": null
        }
      ]
    }
  ]
}

---

## EXAMPLE OUTPUT (single question, no parts)

[
  {
    "question_id": "Q3.1",
    "class": 11,
    "subject": "Physics",
    "chapter_number": 3,
    "chapter_name": "Motion in a Straight Line",
    "question_type": "numerical",
    "question_text": "A car travels 30 km at 40 km/h and next 30 km at 60 km/h. Find average speed.",
    "question_latex": null,
    "mcq_options": [],
    "correct_option": null,
    "question_image": {
      "present": true,
      "description": "Diagram showing particle motion with acceleration vector",
      "latex_extracted": null,
      "cloudinary_url": "https://res.cloudinary.com/demo/image/upload/v1/diagram_velocity.png",
    },
    "parts": [],
    "answer_text": "The average speed is calculated using total distance over total time.",
    "answer_latex": "v_{avg} = \\frac{2 v_1 v_2}{v_1 + v_2} = \\frac{2 \\times 40 \\times 60}{40 + 60} = 48 \\text{ km/h}",
    "steps": [
      {
        "step_number": 1,
        "description": "Identify that both distances are equal, so use harmonic mean formula",
        "expression": "v_{avg} = \\frac{2 v_1 v_2}{v_1 + v_2}"
      },
      {
        "step_number": 2,
        "description": "Substitute values",
        "expression": "v_{avg} = \\frac{2 \\times 40 \\times 60}{40 + 60} = 48"
      }
    ],
    "final_answer": "Therefore, the average speed of the car is 48 km/h.",
    "answer_image": {
      "present": false,
      "description": null,
      "cloudinary_url": null
    },
    "difficulty_rating": 2,
    "difficulty_reasoning": "Requires knowing harmonic mean formula and single substitution.",
    "topics_tags": ["Average Speed", "Uniform Motion", "Kinematics"],
    "confidence_score": 0.97,
    "confidence_issues": []
  }
]"""

SYSTEM_PROMPT += """

---

## MULTI-IMAGE RULES (IMPORTANT)

- A question or answer can contain multiple images.
- Prefer returning `question_images` and `answer_images` arrays at top-level.
- For each part in `parts`, also prefer `question_images` and `answer_images` arrays.
- Keep legacy `question_image` / `answer_image` only if needed; arrays are preferred.
- For each image object include:
  - `present`
  - `description`
  - `cloudinary_url`
  - `latex_extracted`
  - `source_page_number` (from input page/image metadata when available)
  - `source_image_index` (from input image_index when available)
  - `ownership` (`question` or `answer`)
- Use input image hints (`role_hint`, `role_hint_reason`, `context_above`, `context_below`)
  to assign ownership. Do not place answer-only diagrams under `question_images`.
- If ownership is uncertain, keep confidence lower and mention uncertainty in `confidence_issues`.
"""

_OPENAI_CLIENT = None
_OPENAI_INIT_ATTEMPTED = False
_DEBUG_PATH_ENV = 'LLM_PARSER_DEBUG_PATH'
_DEFAULT_DEBUG_FILE = 'llm_parser_debug.jsonl'


def _get_openai_client():
    """
    Lazily initialize OpenAI client.

    Returns:
        OpenAI client instance or None if unavailable.
    """
    global _OPENAI_CLIENT, _OPENAI_INIT_ATTEMPTED
    if _OPENAI_INIT_ATTEMPTED:
        return _OPENAI_CLIENT

    _OPENAI_INIT_ATTEMPTED = True
    try:
        from openai import OpenAI

        _OPENAI_CLIENT = OpenAI()
    except Exception:
        logger.exception('Failed to initialize OpenAI client')
        _OPENAI_CLIENT = None

    return _OPENAI_CLIENT


# def _extract_json_array(raw_text: str) -> str:
#     """
#     Extract a JSON array slice from model output text.
#     """
#     text = (raw_text or '').strip()
#     if not text:
#         return text

#     if text.startswith('```'):
#         text = text.strip('`').strip()

#     first_bracket = text.find('[')
#     last_bracket = text.rfind(']')
#     if first_bracket == -1 or last_bracket == -1 or last_bracket < first_bracket:
#         return text
#     return text[first_bracket:last_bracket + 1]
def _extract_json_array(raw_text: str) -> str:
    """
    Extract a JSON array slice from model output text.
    """
    text = (raw_text or '').strip()
    if not text:
        return text

    # Properly strip markdown code fences
    if text.startswith('```'):
        lines = text.splitlines()
        # Remove first line (```json or ```) 
        lines = lines[1:]
        # Remove last line if it's just ```
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()

    first_bracket = text.find('[')
    last_bracket = text.rfind(']')
    if first_bracket == -1 or last_bracket == -1 or last_bracket < first_bracket:
        return text
    return text[first_bracket:last_bracket + 1]


def _get_debug_file_path() -> str:
    """
    Resolve the JSONL debug file path used for LLM batch diagnostics.
    """
    configured = (os.getenv(_DEBUG_PATH_ENV) or '').strip()
    if configured:
        return configured
    return os.path.join(os.getcwd(), _DEFAULT_DEBUG_FILE)


def _write_debug_record(record: dict) -> None:
    """
    Append one diagnostics record to local JSONL debug file.
    """
    try:
        debug_path = _get_debug_file_path()
        debug_dir = os.path.dirname(debug_path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        with open(debug_path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        logger.exception('Failed to write LLM parser debug record')


def _record_batch_diagnostic(
    reason: str,
    batch_pages: list[dict],
    payload: dict,
    raw_response: Optional[str] = None,
    errors: Optional[list[str]] = None,
) -> None:
    """
    Record detailed diagnostics for one failed or suspicious batch.
    """
    page_numbers = [
        page.get('page_number')
        for page in batch_pages
        if isinstance(page, dict)
    ]
    record = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'reason': reason,
        'page_numbers': page_numbers,
        'errors': errors or [],
        'payload': payload,
        'raw_response': raw_response,
    }
    _write_debug_record(record)


def _safe_float(value, default: float = 0.0) -> float:
    """
    Convert a value to float and clamp to [0.0, 1.0] when applicable.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, parsed))


def _safe_int(value, default: int = 0) -> int:
    """
    Convert a value to integer without raising.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_ANSWER_OWNER_PATTERN = re.compile(
    r'\b(answer|ans\.?|solution|soln|hence|therefore|final answer)\b',
    flags=re.IGNORECASE,
)
_QUESTION_OWNER_PATTERN = re.compile(
    r'\b(question|ques\.?|exercise|given|problem statement)\b',
    flags=re.IGNORECASE,
)


def _normalize_owner_value(value: Optional[str]) -> str:
    """
    Normalize image ownership aliases to question/answer/unknown.
    """
    if value is None:
        return 'unknown'

    normalized = str(value).strip().lower()
    if not normalized:
        return 'unknown'

    if normalized in {'question', 'q', 'prompt', 'problem'}:
        return 'question'
    if normalized in {'answer', 'ans', 'a', 'solution', 'soln'}:
        return 'answer'
    return 'unknown'


def _infer_owner_from_text(text: Optional[str]) -> str:
    """
    Infer ownership from descriptive text when explicit ownership is missing.
    """
    if not text:
        return 'unknown'
    answer_hits = len(_ANSWER_OWNER_PATTERN.findall(text))
    question_hits = len(_QUESTION_OWNER_PATTERN.findall(text))
    if answer_hits > question_hits:
        return 'answer'
    if question_hits > answer_hits:
        return 'question'
    return 'unknown'


def _normalize_image_object(image_obj: dict, fallback_owner: str) -> Optional[dict]:
    """
    Normalize one image object while preserving ownership/source metadata.
    """
    if not isinstance(image_obj, dict):
        return None

    description = image_obj.get('description')
    cloudinary_url = image_obj.get('cloudinary_url') or image_obj.get('image_url')
    latex_extracted = image_obj.get('latex_extracted') or image_obj.get('pix2tex_latex')

    source_page_number = _safe_int(
        image_obj.get('source_page_number', image_obj.get('page_number')),
        0,
    )
    if source_page_number <= 0:
        source_page_number = None

    source_image_index = _safe_int(
        image_obj.get('source_image_index', image_obj.get('image_index')),
        -1,
    )
    if source_image_index < 0:
        source_image_index = None

    explicit_owner_fields = [
        image_obj.get('ownership'),
        image_obj.get('belongs_to'),
        image_obj.get('image_role'),
        image_obj.get('role_hint'),
        image_obj.get('source_type'),
    ]
    ownership = 'unknown'
    for candidate in explicit_owner_fields:
        parsed = _normalize_owner_value(candidate)
        if parsed != 'unknown':
            ownership = parsed
            break

    if ownership == 'unknown':
        ownership = _infer_owner_from_text(str(description or ''))
    if ownership == 'unknown':
        ownership = _normalize_owner_value(fallback_owner)

    present_raw = image_obj.get('present')
    has_content = bool(cloudinary_url or description or latex_extracted)
    present = bool(present_raw) if present_raw is not None else has_content

    if not present and not has_content:
        return None

    normalized_image = {
        'present': present,
        'description': description,
        'cloudinary_url': cloudinary_url,
        'latex_extracted': latex_extracted,
        'source_page_number': source_page_number,
        'source_image_index': source_image_index,
        'ownership': ownership,
    }
    role_hint_reason = image_obj.get('role_hint_reason')
    if role_hint_reason:
        normalized_image['ownership_reason'] = role_hint_reason
    return normalized_image


def _dedupe_images(images: list[dict]) -> list[dict]:
    """
    Remove duplicate images while preserving insertion order.
    """
    deduped: list[dict] = []
    seen: set[tuple] = set()
    for image in images:
        if not isinstance(image, dict):
            continue
        key = (
            image.get('cloudinary_url') or '',
            image.get('source_page_number'),
            image.get('source_image_index'),
            image.get('description') or '',
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(image)
    return deduped


def _empty_image_placeholder() -> dict:
    """
    Backward-compatible empty image object.
    """
    return {
        'present': False,
        'description': None,
        'cloudinary_url': None,
        'latex_extracted': None,
        'source_page_number': None,
        'source_image_index': None,
        'ownership': 'unknown',
    }


def _collect_images(candidate, fallback_owner: str) -> list[dict]:
    """
    Normalize either a single image object or a list of image objects.
    """
    collected: list[dict] = []
    if isinstance(candidate, list):
        for item in candidate:
            normalized = _normalize_image_object(item, fallback_owner)
            if normalized:
                collected.append(normalized)
        return collected

    normalized = _normalize_image_object(candidate, fallback_owner)
    if normalized:
        collected.append(normalized)
    return collected


def _normalize_image_collections(container: dict) -> tuple[list[dict], list[dict]]:
    """
    Normalize image fields for a question/part and return question/answer buckets.
    """
    raw_images: list[dict] = []
    raw_images.extend(_collect_images(container.get('question_images'), 'question'))
    raw_images.extend(_collect_images(container.get('answer_images'), 'answer'))
    raw_images.extend(_collect_images(container.get('question_image'), 'question'))
    raw_images.extend(_collect_images(container.get('answer_image'), 'answer'))
    raw_images.extend(_collect_images(container.get('images'), 'unknown'))

    question_images: list[dict] = []
    answer_images: list[dict] = []

    for image in raw_images:
        owner = _normalize_owner_value(image.get('ownership'))
        if owner == 'answer':
            answer_images.append(image)
        else:
            # Default ambiguous ownership to question bucket for determinism.
            question_images.append(image)

    return _dedupe_images(question_images), _dedupe_images(answer_images)


def _resolve_image_hint_owner(image: dict, image_hints: Optional[dict]) -> str:
    """
    Resolve ownership from extracted-image hints using url or source tuple.
    """
    if not image_hints:
        return 'unknown'

    by_url = image_hints.get('by_url') or {}
    by_source = image_hints.get('by_source') or {}

    cloudinary_url = image.get('cloudinary_url')
    if cloudinary_url and cloudinary_url in by_url:
        return _normalize_owner_value(by_url.get(cloudinary_url))

    source_key = (
        image.get('source_page_number'),
        image.get('source_image_index'),
    )
    if source_key in by_source:
        return _normalize_owner_value(by_source.get(source_key))
    return 'unknown'


def _apply_image_hints(
    question_images: list[dict],
    answer_images: list[dict],
    image_hints: Optional[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Re-bucket images based on deterministic extraction hints.
    """
    if not image_hints:
        return question_images, answer_images

    rebucket_question: list[dict] = []
    rebucket_answer: list[dict] = []
    for image in (question_images + answer_images):
        hinted_owner = _resolve_image_hint_owner(image, image_hints)
        if hinted_owner in {'question', 'answer'}:
            image['ownership'] = hinted_owner

        if _normalize_owner_value(image.get('ownership')) == 'answer':
            rebucket_answer.append(image)
        else:
            rebucket_question.append(image)

    return _dedupe_images(rebucket_question), _dedupe_images(rebucket_answer)


def _first_image_or_empty(images: list[dict]) -> dict:
    """
    Return first image or an empty placeholder for backward compatibility.
    """
    if images:
        return images[0]
    return _empty_image_placeholder()


def _normalize_parts(parts_raw, image_hints: Optional[dict] = None) -> list[dict]:
    """
    Normalize part objects and support multi-image fields per part.
    """
    if not isinstance(parts_raw, list):
        return []

    parts: list[dict] = []
    for part in parts_raw:
        if not isinstance(part, dict):
            continue
        normalized_part = dict(part)
        normalized_part['steps'] = normalized_part.get('steps') or []

        part_question_images, part_answer_images = _normalize_image_collections(normalized_part)
        part_question_images, part_answer_images = _apply_image_hints(
            part_question_images,
            part_answer_images,
            image_hints,
        )
        normalized_part['question_images'] = part_question_images
        normalized_part['answer_images'] = part_answer_images
        normalized_part['question_image'] = _first_image_or_empty(part_question_images)
        normalized_part['answer_image'] = _first_image_or_empty(part_answer_images)
        parts.append(normalized_part)
    return parts


def _is_mcq_text(text: str) -> bool:
    """
    Heuristically detect MCQ-like option patterns in question text.
    """
    if not text:
        return False

    numbered = re.findall(r'\(\s*[1-9]\s*\)', text)
    lettered = re.findall(r'\(\s*[A-Da-d]\s*\)', text)
    if len(numbered) + len(lettered) >= 2:
        return True

    inline_option_like = re.findall(r'(?<!\w)(?:[1-9]|[A-Da-d])[\)\.:]', text)
    return len(inline_option_like) >= 2


def _extract_mcq_options(text: str) -> list[dict]:
    """
    Extract option labels and texts from question text.
    """
    if not text:
        return []

    matches = list(re.finditer(r'\(\s*(?P<label>[1-9]|[A-Da-d])\s*\)', text))
    if len(matches) < 2:
        return []

    options: list[dict] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if (index + 1) < len(matches) else len(text)
        option_text = text[start:end].strip(' \t\n\r:;-')
        if not option_text:
            continue
        label = f"({str(match.group('label')).upper()})"
        options.append(
            {
                'option_label': label,
                'option_text': option_text,
            }
        )
    return options


def _extract_correct_option(answer_text: Optional[str]) -> Optional[str]:
    """
    Extract MCQ correct option from answer text when possible.
    """
    if not answer_text:
        return None

    text = str(answer_text)
    bracketed = re.findall(r'\(\s*([1-9]|[A-Da-d])\s*\)', text)
    if bracketed:
        return f"({str(bracketed[0]).upper()})"

    direct = re.search(
        r'(?:answer\s*(?:is|:)?|correct\s*option\s*(?:is|:)?|option\s*)([1-9]|[A-Da-d])\b',
        text,
        flags=re.IGNORECASE,
    )
    if direct:
        return f"({str(direct.group(1)).upper()})"

    return None


def _normalize_question(
    question: dict,
    metadata: dict,
    fallback_index: int,
    image_hints: Optional[dict] = None,
) -> dict:
    """
    Normalize a question object to expected keys and defaults.
    """
    chapter_number = metadata.get('chapter_number')
    default_qid = f"Q{chapter_number}.{fallback_index}"
    question_id = question.get('question_id') or default_qid
    confidence_score = _safe_float(question.get('confidence_score', 0.0))

    normalized = dict(question)
    normalized['question_id'] = str(question_id)
    normalized['class'] = _safe_int(normalized.get('class', metadata.get('class', 0)), 0)
    normalized['subject'] = normalized.get('subject') or metadata.get('subject')
    normalized['chapter_number'] = _safe_int(
        normalized.get('chapter_number', metadata.get('chapter_number', 0)) or 0
    )
    normalized['chapter_name'] = normalized.get('chapter_name') or metadata.get('chapter_name')
    normalized['steps'] = normalized.get('steps') or []
    normalized['topics_tags'] = normalized.get('topics_tags') or []
    normalized['confidence_issues'] = normalized.get('confidence_issues') or []
    normalized['confidence_score'] = confidence_score
    normalized['mcq_options'] = normalized.get('mcq_options') or []
    normalized['correct_option'] = normalized.get('correct_option')

    question_images, answer_images = _normalize_image_collections(normalized)
    question_images, answer_images = _apply_image_hints(
        question_images,
        answer_images,
        image_hints,
    )
    normalized['question_images'] = question_images
    normalized['answer_images'] = answer_images
    normalized['question_image'] = _first_image_or_empty(question_images)
    normalized['answer_image'] = _first_image_or_empty(answer_images)
    normalized['parts'] = _normalize_parts(normalized.get('parts'), image_hints=image_hints)

    question_text = str(normalized.get('question_text') or '')
    question_type = str(normalized.get('question_type') or '').strip().lower()
    is_mcq = (question_type == 'mcq') or _is_mcq_text(question_text)

    if is_mcq:
        normalized['question_type'] = 'mcq'
        if not normalized['mcq_options']:
            normalized['mcq_options'] = _extract_mcq_options(question_text)
        if not normalized.get('correct_option'):
            normalized['correct_option'] = _extract_correct_option(normalized.get('answer_text'))
    else:
        normalized['mcq_options'] = []
        normalized['correct_option'] = None

    return normalized


def _call_llm_batch(payload: dict, max_attempts: int = 3) -> tuple[Optional[str], list[str]]:
    """
    Execute one LLM call with retry/backoff on API failures.

    Returns:
        Tuple of:
            - raw response text on success, otherwise None
            - list of failure notes across attempts
    """
    client = _get_openai_client()
    if client is None:
        return None, ['OpenAI client unavailable']

    user_content = json.dumps(payload, ensure_ascii=False)
    failure_notes: list[str] = []

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': user_content},
                ],
                temperature=0,
            )
            if not response.choices:
                logger.error('LLM returned no choices for payload batch')
                failure_notes.append(f'attempt {attempt}: no choices in response')
                return None, failure_notes
            content = response.choices[0].message.content
            return content or '', failure_notes
        except Exception as exc:
            failure_notes.append(
                f'attempt {attempt}: {exc.__class__.__name__}: {str(exc)}'
            )
            if attempt >= max_attempts:
                logger.exception('LLM batch failed after %s attempts', max_attempts)
                return None, failure_notes
            sleep_seconds = 2 ** attempt
            logger.warning(
                'LLM API failure on attempt %s/%s. Retrying in %s seconds.',
                attempt,
                max_attempts,
                sleep_seconds,
                exc_info=True,
            )
            time.sleep(sleep_seconds)

    return None, failure_notes


# def _build_batch_payload(batch_pages: list[dict], metadata: dict) -> dict:
def _build_batch_payload(batch_pages: list[dict], metadata: dict, question_start_index=1) -> dict:
    """
    Build LLM payload for a page batch.
    """
    payload_pages = []
    for page in batch_pages:
        if not isinstance(page, dict):
            continue
        images = page.get('images') or []
        payload_images = []
        latex_blocks = []

        for image in images:
            if not isinstance(image, dict):
                continue
            latex = image.get('pix2tex_latex')
            if latex:
                latex_blocks.append(latex)

            payload_images.append(
                {
                    'image_index': image.get('image_index'),
                    'source_image_index': image.get('image_index'),
                    'source_page_number': image.get('page_number') or page.get('page_number'),
                    'position': image.get('position'),
                    'cloudinary_url': image.get('cloudinary_url'),
                    'pix2tex_latex': latex,
                    'ocr_text': image.get('ocr_text'),
                    'is_mathematical': bool(image.get('is_mathematical')),
                    'context_above': image.get('context_above'),
                    'context_below': image.get('context_below'),
                    'role_hint': image.get('role_hint'),
                    'role_hint_reason': image.get('role_hint_reason'),
                    'bbox': image.get('bbox'),
                }
            )

        payload_pages.append(
            {
                'page_number': page.get('page_number'),
                'text': page.get('text') or '',
                'latex_blocks': latex_blocks,
                'images': payload_images,
            }
        )

    # return {'metadata': metadata, 'pages': payload_pages}
    return {
        'metadata': metadata,
        'pages': payload_pages,
        'indexing_hint': {
            'question_index_start': question_start_index,
            'instruction': (
                'Start question_id numbering from this index. '
                'Do not restart numbering from 1 mid-document.'
            ),
        },
    }

def _is_hex_digit(value: str) -> bool:
    """
    Return True if char is a hexadecimal digit.
    """
    return value.lower() in '0123456789abcdef'


def _has_valid_unicode_escape(text: str, slash_index: int) -> bool:
    """
    Check whether a backslash-u escape at index is valid JSON unicode escape.
    """
    if slash_index + 5 >= len(text):
        return False
    if text[slash_index + 1] != 'u':
        return False
    code = text[slash_index + 2:slash_index + 6]
    return len(code) == 4 and all(_is_hex_digit(ch) for ch in code)


def _sanitize_latex_escapes(raw_text: str) -> str:
    """
    Repair invalid backslashes inside JSON string literals.

    This is conservative and state-aware:
    - preserves already valid escaped sequences
    - doubles invalid single backslashes (common from LaTeX like \frac, \epsilon)
    - avoids corrupting sequences such as `\\,` and `\\text`
    """
    result: list[str] = []
    in_string = False
    escaped = False
    i = 0
    length = len(raw_text)

    while i < length:
        char = raw_text[i]

        if not in_string:
            result.append(char)
            if char == '"':
                in_string = True
                escaped = False
            i += 1
            continue

        if escaped:
            result.append(char)
            escaped = False
            i += 1
            continue

        if char == '"':
            result.append(char)
            in_string = False
            i += 1
            continue

        if char != '\\':
            result.append(char)
            i += 1
            continue

        next_char = raw_text[i + 1] if (i + 1) < length else ''
        if next_char in {'"', '\\', '/'}:
            # Keep valid JSON escapes that are common in model output.
            result.append(char)
            escaped = True
            i += 1
            continue

        if next_char == 'u' and _has_valid_unicode_escape(raw_text, i):
            result.append(char)
            escaped = True
            i += 1
            continue

        # Invalid escape inside JSON string, likely LaTeX command.
        result.append('\\\\')
        i += 1

    return ''.join(result)


def _extract_top_level_json_objects(array_text: str) -> list[str]:
    """
    Extract top-level JSON object strings from an array-like text.
    """
    objects: list[str] = []
    in_string = False
    escaped = False
    depth = 0
    object_start: Optional[int] = None

    for index, char in enumerate(array_text):
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == '\\':
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == '{':
            if depth == 0:
                object_start = index
            depth += 1
            continue
        if char == '}':
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and object_start is not None:
                objects.append(array_text[object_start:index + 1])
                object_start = None

    return objects

def _parse_batch_response(raw_response: str) -> Optional[list[dict]]:
    """
    Parse and validate one batch response.
    """
    json_text = _extract_json_array(raw_response)
    sanitized = _sanitize_latex_escapes(json_text)

    parse_errors: list[str] = []
    for candidate_name, candidate in [('raw', json_text), ('sanitized', sanitized)]:
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, list):
                logger.error('LLM response was not a JSON array')
                return None
            return [item for item in parsed if isinstance(item, dict)]
        except json.JSONDecodeError as exc:
            parse_errors.append(f'{candidate_name}: {exc}')
            if candidate_name == 'raw':
                logger.warning(
                    'Primary JSON parse failed (%s), trying fallback sanitization',
                    exc,
                )

    # Last resort: salvage top-level objects so one malformed entry does not drop whole batch.
    salvaged: list[dict] = []
    for object_text in _extract_top_level_json_objects(sanitized):
        try:
            item = json.loads(object_text)
        except json.JSONDecodeError:
            try:
                item = json.loads(_sanitize_latex_escapes(object_text))
            except json.JSONDecodeError:
                continue
        if isinstance(item, dict):
            salvaged.append(item)

    if salvaged:
        logger.warning(
            'Recovered %s question objects from malformed batch JSON.',
            len(salvaged),
        )
        return salvaged

    raise json.JSONDecodeError('; '.join(parse_errors), sanitized, 0)


def _deduplicate_questions(questions: list[dict]) -> list[dict]:
    """
    Deduplicate by question_id and keep record with higher confidence_score.
    """
    best_by_id: dict[str, dict] = {}

    for question in questions:
        question_id = str(question.get('question_id') or '').strip()
        if not question_id:
            continue

        existing = best_by_id.get(question_id)
        if existing is None:
            best_by_id[question_id] = question
            continue

        new_score = _safe_float(question.get('confidence_score', 0.0))
        old_score = _safe_float(existing.get('confidence_score', 0.0))
        if new_score > old_score:
            best_by_id[question_id] = question

    return list(best_by_id.values())


def _build_image_hints_map(pages_data: list[dict]) -> dict:
    """
    Build deterministic image ownership hints from extracted page-image metadata.
    """
    by_url: dict[str, str] = {}
    by_source: dict[tuple[int, int], str] = {}

    for page in pages_data or []:
        if not isinstance(page, dict):
            continue
        page_number = _safe_int(page.get('page_number'), 0)
        for image in page.get('images') or []:
            if not isinstance(image, dict):
                continue
            owner = _normalize_owner_value(image.get('role_hint'))
            if owner == 'unknown':
                continue

            cloudinary_url = image.get('cloudinary_url')
            if cloudinary_url and cloudinary_url not in by_url:
                by_url[str(cloudinary_url)] = owner

            source_index = _safe_int(image.get('image_index'), -1)
            if page_number > 0 and source_index >= 0:
                source_key = (page_number, source_index)
                if source_key not in by_source:
                    by_source[source_key] = owner

    return {'by_url': by_url, 'by_source': by_source}


_QUESTION_MARKER_PATTERN = re.compile(
    r'\bquestion\s+\d+\.\d+\s*:?',
    flags=re.IGNORECASE,
)


def _estimate_question_markers(batch_pages: list[dict]) -> int:
    """
    Estimate how many questions should exist in a batch from OCR/text markers.
    """
    total = 0
    for page in batch_pages:
        if not isinstance(page, dict):
            continue
        text = str(page.get('text') or '')
        total += len(_QUESTION_MARKER_PATTERN.findall(text))
    return total


def _recover_batch_by_single_page_calls(
    batch_pages: list[dict],
    metadata: dict,
    question_start_index: int,
) -> list[dict]:
    """
    Retry a failed/underperforming batch by parsing one page at a time.
    """
    recovered: list[dict] = []
    next_question_index = question_start_index

    for page in batch_pages:
        if not isinstance(page, dict):
            continue
        single_payload = _build_batch_payload(
            [page],
            metadata,
            question_start_index=next_question_index,
        )
        raw_response, _ = _call_llm_batch(single_payload, max_attempts=2)
        if raw_response is None:
            continue
        try:
            page_questions = _parse_batch_response(raw_response)
        except json.JSONDecodeError:
            continue
        if not page_questions:
            continue
        recovered.extend(page_questions)
        next_question_index += len(page_questions)

    return recovered


def parse_pages_with_llm(
    pages_data: list[dict],
    metadata: dict,
    batch_size: int = 3,
) -> list[dict]:
    """
    Parse page content using GPT and return a flat list of question dictionaries.

    This function never raises. Failed batches are logged and skipped.
    """
    if not pages_data:
        return []

    try:
        max_batch_size = max(1, min(int(batch_size), 5))
    except (TypeError, ValueError):
        max_batch_size = 3
    all_questions: list[dict] = []
    fallback_index = 1
    image_hints = _build_image_hints_map(pages_data)

    for start_index in range(0, len(pages_data), max_batch_size):
        try:
            batch_pages = pages_data[start_index:start_index + max_batch_size]
            expected_markers = _estimate_question_markers(batch_pages)
            payload = _build_batch_payload(batch_pages, metadata, question_start_index=fallback_index,)

            raw_response, failure_notes = _call_llm_batch(payload, max_attempts=3)
            if raw_response is None:
                logger.error(
                    'Skipping failed LLM batch for pages: %s',
                    [
                        page.get('page_number')
                        for page in batch_pages
                        if isinstance(page, dict)
                    ],
                )
                _record_batch_diagnostic(
                    reason='api_failure_after_retries',
                    batch_pages=batch_pages,
                    payload=payload,
                    raw_response=None,
                    errors=failure_notes,
                )
                continue

            try:
                batch_questions = _parse_batch_response(raw_response)
            except json.JSONDecodeError:
                recovered = _recover_batch_by_single_page_calls(
                    batch_pages=batch_pages,
                    metadata=metadata,
                    question_start_index=fallback_index,
                )
                if recovered:
                    logger.warning(
                        'Recovered %s questions via single-page fallback for pages %s.',
                        len(recovered),
                        [
                            page.get('page_number')
                            for page in batch_pages
                            if isinstance(page, dict)
                        ],
                    )
                    batch_questions = recovered
                else:
                    logger.error(
                        'Skipping batch due to JSON parse failure. Raw response: %s',
                        raw_response,
                    )
                    _record_batch_diagnostic(
                        reason='json_parse_failure',
                        batch_pages=batch_pages,
                        payload=payload,
                        raw_response=raw_response,
                        errors=['JSONDecodeError while parsing model output'],
                    )
                    continue
            except Exception as exc:
                logger.exception('Unexpected response parsing error. Skipping batch.')
                _record_batch_diagnostic(
                    reason='response_parse_unexpected_exception',
                    batch_pages=batch_pages,
                    payload=payload,
                    raw_response=raw_response,
                    errors=[f'{exc.__class__.__name__}: {str(exc)}'],
                )
                continue

            if not batch_questions:
                _record_batch_diagnostic(
                    reason='empty_batch_result',
                    batch_pages=batch_pages,
                    payload=payload,
                    raw_response=raw_response,
                    errors=['Model output parsed but produced no question objects'],
                )
                continue

            if expected_markers > 0 and len(batch_questions) + 1 < expected_markers:
                recovered = _recover_batch_by_single_page_calls(
                    batch_pages=batch_pages,
                    metadata=metadata,
                    question_start_index=fallback_index,
                )
                if len(recovered) > len(batch_questions):
                    logger.warning(
                        'Batch under-extracted questions (%s vs markers=%s). '
                        'Using single-page recovery result (%s) for pages %s.',
                        len(batch_questions),
                        expected_markers,
                        len(recovered),
                        [
                            page.get('page_number')
                            for page in batch_pages
                            if isinstance(page, dict)
                        ],
                    )
                    batch_questions = recovered

            for question in batch_questions:
                normalized = _normalize_question(
                    question,
                    metadata,
                    fallback_index,
                    image_hints=image_hints,
                )
                all_questions.append(normalized)
                fallback_index += 1
        except Exception as exc:
            logger.exception('Unexpected batch handling error. Skipping batch.')
            _record_batch_diagnostic(
                reason='batch_unexpected_exception',
                batch_pages=batch_pages if 'batch_pages' in locals() else [],
                payload=payload if 'payload' in locals() else {},
                raw_response=None,
                errors=[f'{exc.__class__.__name__}: {str(exc)}'],
            )
            continue

    return _deduplicate_questions(all_questions)
