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


def _extract_json_array(raw_text: str) -> str:
    """
    Extract a JSON array slice from model output text.
    """
    text = (raw_text or '').strip()
    if not text:
        return text

    if text.startswith('```'):
        text = text.strip('`').strip()

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


def _normalize_question(question: dict, metadata: dict, fallback_index: int) -> dict:
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
    normalized['parts'] = normalized.get('parts') or []
    normalized['steps'] = normalized.get('steps') or []
    normalized['topics_tags'] = normalized.get('topics_tags') or []
    normalized['confidence_issues'] = normalized.get('confidence_issues') or []
    normalized['confidence_score'] = confidence_score
    normalized['mcq_options'] = normalized.get('mcq_options') or []
    normalized['correct_option'] = normalized.get('correct_option')

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


def _build_batch_payload(batch_pages: list[dict], metadata: dict) -> dict:
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
                    'position': image.get('position'),
                    'cloudinary_url': image.get('cloudinary_url'),
                    'pix2tex_latex': latex,
                    'ocr_text': image.get('ocr_text'),
                    'is_mathematical': bool(image.get('is_mathematical')),
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

    return {'metadata': metadata, 'pages': payload_pages}


def _parse_batch_response(raw_response: str) -> Optional[list[dict]]:
    """
    Parse and validate one batch response.
    """
    json_text = _extract_json_array(raw_response)
    parsed = json.loads(json_text)
    if not isinstance(parsed, list):
        logger.error('LLM response was not a JSON array')
        return None
    return [item for item in parsed if isinstance(item, dict)]


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

    for start_index in range(0, len(pages_data), max_batch_size):
        try:
            batch_pages = pages_data[start_index:start_index + max_batch_size]
            payload = _build_batch_payload(batch_pages, metadata)

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

            for question in batch_questions:
                normalized = _normalize_question(question, metadata, fallback_index)
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
