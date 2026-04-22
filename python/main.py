import sys
import json
import os
import logging
from contextlib import contextmanager
from time import perf_counter

from dotenv import load_dotenv

# from database.db import insert_questions
from parser.extractor import extract_text_from_pdf, extract_text_per_page
from parser.image_extractor import extract_images_per_page
from parser.llm_parser import parse_pages_with_llm

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """
    Configure application logging to stderr.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s [%(name)s] %(message)s',
        stream=sys.stderr,
        force=True,
    )


def _parse_cli_args(argv: list[str]) -> dict:
    """
    Parse and validate CLI arguments.
    """
    if len(argv) < 7:
        raise ValueError(
            'Usage: main.py <pdf_path> <class> <subject> <chapter_num> <chapter_name> <output_path>'
        )

    pdf_path = argv[1]
    class_num = int(argv[2])
    subject = argv[3]
    chapter_num = int(argv[4])
    chapter_name = argv[5]
    output_path = argv[6]

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    return {
        'pdf_path': pdf_path,
        'class_num': class_num,
        'subject': subject,
        'chapter_num': chapter_num,
        'chapter_name': chapter_name,
        'output_path': output_path,
    }


def _format_log_details(details: dict[str, object]) -> str:
    """
    Convert keyword details into a stable 'k=v' comma-separated string.
    """
    if not details:
        return ''
    return ', '.join(f'{key}={details[key]}' for key in sorted(details.keys()))


@contextmanager
def _timed_step(step_name: str, **details: object):
    """
    Log a START/END marker with elapsed time for long-running steps.
    """
    detail_str = _format_log_details(details)
    if detail_str:
        logger.info('START: %s (%s)', step_name, detail_str)
    else:
        logger.info('START: %s', step_name)

    started_at = perf_counter()
    completed = False
    try:
        yield
        completed = True
    finally:
        elapsed = perf_counter() - started_at
        status = 'END' if completed else 'ABORTED'
        logger.info('%s: %s (%.2fs)', status, step_name, elapsed)


def _merge_page_text_into_images(
    pages_with_images: list[dict],
    page_text_entries: list[dict],
) -> list[dict]:
    """
    Merge per-page text into extracted image metadata structure.
    """
    merged_map: dict[int, dict] = {}

    for page in pages_with_images or []:
        page_number = int(page.get('page_number') or 0)
        if page_number <= 0:
            continue
        merged_map[page_number] = {
            'page_number': page_number,
            'is_image_only_page': bool(page.get('is_image_only_page')),
            'images': page.get('images') or [],
            'text': '',
        }

    for entry in page_text_entries or []:
        page_number = int(entry.get('page_number') or 0)
        if page_number <= 0:
            continue
        text = entry.get('text') or ''

        if page_number not in merged_map:
            merged_map[page_number] = {
                'page_number': page_number,
                'is_image_only_page': False,
                'images': [],
                'text': text,
            }
            continue

        merged_map[page_number]['text'] = text
        if len(text.strip()) < 50 and len(merged_map[page_number]['images']) > 0:
            merged_map[page_number]['is_image_only_page'] = True

    return [merged_map[key] for key in sorted(merged_map.keys())]


def _extract_flagged_question_ids(questions: list[dict]) -> list[str]:
    """
    Return question IDs where confidence_score is below 0.5.
    """
    flagged = []
    for question in questions:
        if not isinstance(question, dict):
            continue
        question_id = question.get('question_id')
        if not question_id:
            continue
        try:
            confidence = float(question.get('confidence_score', 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        if confidence < 0.5:
            flagged.append(str(question_id))
    return flagged

def main():
    """
    Run end-to-end PDF parsing pipeline and emit one JSON summary to stdout.
    """
    _configure_logging()
    with _timed_step('Load environment variables'):
        load_dotenv()

    try:
        with _timed_step('Parse CLI arguments'):
            args = _parse_cli_args(sys.argv)
            pdf_path = args['pdf_path']
            class_num = args['class_num']
            subject = args['subject']
            chapter_num = args['chapter_num']
            chapter_name = args['chapter_name']
            output_path = args['output_path']

        logger.info(
            'Pipeline context: pdf_path=%s, class=%s, subject=%s, chapter_num=%s, chapter_name=%s, output_path=%s',
            pdf_path,
            class_num,
            subject,
            chapter_num,
            chapter_name,
            output_path,
        )

        with _timed_step('Extract full PDF text'):
            try:
                full_text = extract_text_from_pdf(pdf_path)
                logger.info('Extracted full text (%s chars)', len(full_text))
            except Exception:
                logger.exception(
                    'Full-text extraction failed; continuing with per-page extraction only'
                )
                full_text = ''

        with _timed_step('Extract images per page'):
            pages_with_images = extract_images_per_page(
                pdf_path=pdf_path,
                subject=subject,
                chapter_num=chapter_num,
            )
            logger.info('Image extraction pages: %s', len(pages_with_images or []))

        with _timed_step('Extract text per page'):
            page_text_entries = extract_text_per_page(pdf_path)
            logger.info('Text extraction pages: %s', len(page_text_entries or []))

        with _timed_step('Merge text and image page data'):
            merged_pages_data = _merge_page_text_into_images(pages_with_images, page_text_entries)
            logger.info('Merged pages: %s', len(merged_pages_data))

        metadata = {
            'class': class_num,
            'subject': subject,
            'chapter_number': chapter_num,
            'chapter_name': chapter_name,
        }
        batch_size = 3
        with _timed_step(
            'Parse pages with LLM',
            pages=len(merged_pages_data),
            batch_size=batch_size,
        ):
            questions = parse_pages_with_llm(
                pages_data=merged_pages_data,
                metadata=metadata,
                batch_size=batch_size,
            )
            logger.info('Parsed questions: %s', len(questions))

        with _timed_step('Write output JSON', output_path=output_path):
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)

        # with _timed_step('Insert parsed questions into database'):
        #     try:
        #         db_result = insert_questions(questions)
        #         logger.info('Database insert result: %s', db_result)
        #     except Exception:
        #         logger.exception('Database insert failed; continuing without failing process')

        with _timed_step('Extract low-confidence question IDs'):
            flagged = _extract_flagged_question_ids(questions)
            logger.info('Low-confidence question count: %s', len(flagged))

        print(json.dumps({
            "success": True,
            "count": len(questions),
            "outputPath": output_path,
            "flagged": flagged,
            "questions": questions
        }))
        logger.info('Pipeline finished successfully')

    except Exception as e:
        logger.exception('Pipeline failed with an unhandled exception')
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
