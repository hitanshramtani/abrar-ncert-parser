"""Database helpers for storing and querying parsed NCERT questions."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, execute_values

logger = logging.getLogger(__name__)

_SCHEMA_INITIALIZED = False


def _load_db_config() -> Optional[dict]:
    """
    Load PostgreSQL connection settings from environment variables.
    """
    load_dotenv()
    db_name = os.getenv('DB_NAME') or os.getenv('PGDATABASE')
    db_user = os.getenv('DB_USER') or os.getenv('PGUSER')
    db_password = os.getenv('DB_PASSWORD') or os.getenv('PGPASSWORD')
    db_host = os.getenv('DB_HOST') or os.getenv('PGHOST') or 'localhost'
    db_port = os.getenv('DB_PORT') or os.getenv('PGPORT') or '5432'

    if not db_name or not db_user:
        logger.warning('DB_NAME/DB_USER (or PGDATABASE/PGUSER) not configured')
        return None

    return {
        'dbname': db_name,
        'user': db_user,
        'password': db_password,
        'host': db_host,
        'port': db_port,
    }


def _get_connection():
    """
    Create and return a psycopg2 connection using .env configuration.
    """
    config = _load_db_config()
    if config is None:
        return None
    return psycopg2.connect(**config)


def _ensure_schema(conn) -> None:
    """
    Ensure required tables/indexes exist using schema.sql.
    """
    global _SCHEMA_INITIALIZED
    if _SCHEMA_INITIALIZED:
        return

    schema_path = Path(__file__).resolve().parent / 'schema.sql'
    schema_sql = schema_path.read_text(encoding='utf-8')
    with conn.cursor() as cursor:
        cursor.execute(schema_sql)
    conn.commit()
    _SCHEMA_INITIALIZED = True


def _coerce_int(value: Any, default: int = 0) -> int:
    """
    Safely cast value to int.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """
    Safely cast value to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_image_url(image_obj: dict | None) -> Optional[str]:
    """
    Extract Cloudinary URL from an image object.
    """
    if not isinstance(image_obj, dict):
        return None
    url = image_obj.get('cloudinary_url') or image_obj.get('image_url')
    return str(url) if url else None


def _normalize_image_row(
    image_obj: dict | None,
    image_type: str,
    question_db_id: int,
    part_id: Optional[int] = None,
) -> Optional[tuple]:
    """
    Convert an image dictionary to a DB row tuple.
    """
    if not isinstance(image_obj, dict):
        return None

    present = bool(image_obj.get('present'))
    url = _extract_image_url(image_obj)
    description = image_obj.get('description')
    latex_extracted = image_obj.get('latex_extracted') or image_obj.get('pix2tex_latex')

    if not present and not url and not description and not latex_extracted:
        return None

    return (
        question_db_id,
        part_id,
        image_type,
        url,
        description,
        latex_extracted,
    )


def _normalize_steps(steps: list | None) -> list[dict]:
    """
    Normalize step entries to a uniform list of dictionaries.
    """
    if not isinstance(steps, list):
        return []

    normalized = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        normalized.append(
            {
                'step_number': _coerce_int(step.get('step_number', index), index),
                'description': step.get('description'),
                'expression': step.get('expression'),
            }
        )
    return normalized


def insert_questions(questions: list) -> dict:
    """
    Insert a chapter's parsed questions into PostgreSQL.

    Args:
        questions: List of parsed question dictionaries.

    Returns:
        Summary dict with inserted count.
    """
    if not questions:
        return {'inserted_questions': 0}

    conn = _get_connection()
    if conn is None:
        return {'inserted_questions': 0, 'db_configured': False}

    try:
        _ensure_schema(conn)
        inserted_count = 0

        with conn.cursor() as cursor:
            question_rows = []
            for question in questions:
                if not isinstance(question, dict):
                    continue
                question_id = question.get('question_id')
                if not question_id:
                    continue
                question_rows.append(
                    (
                        str(question_id),
                        _coerce_int(question.get('class'), 0),
                        str(question.get('subject') or ''),
                        _coerce_int(question.get('chapter_number'), 0),
                        str(question.get('chapter_name') or ''),
                        question.get('question_type'),
                        question.get('question_text'),
                        question.get('question_latex'),
                        question.get('answer_text'),
                        question.get('answer_latex'),
                        question.get('final_answer'),
                        _coerce_int(question.get('difficulty_rating'), 0),
                        question.get('difficulty_reasoning'),
                        _coerce_float(question.get('confidence_score'), 0.0),
                    )
                )

            if not question_rows:
                return {'inserted_questions': 0}

            upsert_sql = """
                INSERT INTO questions (
                    question_id, class, subject, chapter_number, chapter_name,
                    question_type, question_text, question_latex, answer_text, answer_latex,
                    final_answer, difficulty_rating, difficulty_reasoning, confidence_score
                )
                VALUES %s
                ON CONFLICT (question_id) DO UPDATE SET
                    class = EXCLUDED.class,
                    subject = EXCLUDED.subject,
                    chapter_number = EXCLUDED.chapter_number,
                    chapter_name = EXCLUDED.chapter_name,
                    question_type = EXCLUDED.question_type,
                    question_text = EXCLUDED.question_text,
                    question_latex = EXCLUDED.question_latex,
                    answer_text = EXCLUDED.answer_text,
                    answer_latex = EXCLUDED.answer_latex,
                    final_answer = EXCLUDED.final_answer,
                    difficulty_rating = EXCLUDED.difficulty_rating,
                    difficulty_reasoning = EXCLUDED.difficulty_reasoning,
                    confidence_score = EXCLUDED.confidence_score
                RETURNING id, question_id
            """
            execute_values(cursor, upsert_sql, question_rows, page_size=100)
            id_pairs = cursor.fetchall()
            question_id_map = {question_id: db_id for db_id, question_id in id_pairs}

            if not question_id_map:
                return {'inserted_questions': 0}

            inserted_count = len(question_id_map)
            question_db_ids = list(set(question_id_map.values()))

            cursor.execute(
                'DELETE FROM question_images WHERE question_db_id = ANY(%s)',
                (question_db_ids,),
            )
            cursor.execute(
                'DELETE FROM steps WHERE question_db_id = ANY(%s)',
                (question_db_ids,),
            )
            cursor.execute(
                'DELETE FROM question_parts WHERE question_db_id = ANY(%s)',
                (question_db_ids,),
            )
            cursor.execute(
                'DELETE FROM topics_tags WHERE question_db_id = ANY(%s)',
                (question_db_ids,),
            )

            top_level_step_rows: list[tuple] = []
            top_level_image_rows: list[tuple] = []
            topic_rows: list[tuple] = []
            part_step_rows: list[tuple] = []
            part_image_rows: list[tuple] = []

            for question in questions:
                if not isinstance(question, dict):
                    continue
                question_id = str(question.get('question_id') or '')
                question_db_id = question_id_map.get(question_id)
                if question_db_id is None:
                    continue

                for step in _normalize_steps(question.get('steps')):
                    top_level_step_rows.append(
                        (
                            question_db_id,
                            None,
                            step['step_number'],
                            step['description'],
                            step['expression'],
                        )
                    )

                question_image = _normalize_image_row(
                    question.get('question_image'),
                    'question',
                    question_db_id,
                    None,
                )
                if question_image:
                    top_level_image_rows.append(question_image)

                answer_image = _normalize_image_row(
                    question.get('answer_image'),
                    'answer',
                    question_db_id,
                    None,
                )
                if answer_image:
                    top_level_image_rows.append(answer_image)

                for tag in question.get('topics_tags') or []:
                    if not isinstance(tag, str):
                        continue
                    normalized_tag = tag.strip()
                    if not normalized_tag:
                        continue
                    topic_rows.append((question_db_id, normalized_tag))

                for part in question.get('parts') or []:
                    if not isinstance(part, dict):
                        continue
                    cursor.execute(
                        """
                        INSERT INTO question_parts (
                            question_db_id, part_label, part_text, part_latex,
                            answer_text, answer_latex, final_answer
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            question_db_id,
                            part.get('part_label'),
                            part.get('part_text'),
                            part.get('part_latex'),
                            part.get('answer_text'),
                            part.get('answer_latex'),
                            part.get('final_answer'),
                        ),
                    )
                    part_id = cursor.fetchone()[0]

                    for step in _normalize_steps(part.get('steps')):
                        part_step_rows.append(
                            (
                                question_db_id,
                                part_id,
                                step['step_number'],
                                step['description'],
                                step['expression'],
                            )
                        )

                    part_question_image = _normalize_image_row(
                        part.get('question_image'),
                        'question',
                        question_db_id,
                        part_id,
                    )
                    if part_question_image:
                        part_image_rows.append(part_question_image)

                    part_answer_image = _normalize_image_row(
                        part.get('answer_image'),
                        'answer',
                        question_db_id,
                        part_id,
                    )
                    if part_answer_image:
                        part_image_rows.append(part_answer_image)

            if top_level_step_rows:
                execute_values(
                    cursor,
                    """
                    INSERT INTO steps (question_db_id, part_id, step_number, description, expression)
                    VALUES %s
                    """,
                    top_level_step_rows,
                    page_size=500,
                )

            combined_step_rows = part_step_rows
            if combined_step_rows:
                execute_values(
                    cursor,
                    """
                    INSERT INTO steps (question_db_id, part_id, step_number, description, expression)
                    VALUES %s
                    """,
                    combined_step_rows,
                    page_size=500,
                )

            image_rows = top_level_image_rows + part_image_rows
            if image_rows:
                execute_values(
                    cursor,
                    """
                    INSERT INTO question_images (
                        question_db_id, part_id, image_type, cloudinary_url, description, latex_extracted
                    )
                    VALUES %s
                    """,
                    image_rows,
                    page_size=200,
                )

            if topic_rows:
                execute_values(
                    cursor,
                    """
                    INSERT INTO topics_tags (question_db_id, tag)
                    VALUES %s
                    ON CONFLICT (question_db_id, tag) DO NOTHING
                    """,
                    topic_rows,
                    page_size=500,
                )

        conn.commit()
        return {'inserted_questions': inserted_count, 'db_configured': True}
    except Exception:
        conn.rollback()
        logger.exception('Failed to insert questions into database')
        raise
    finally:
        conn.close()


def _fetch_steps(cursor, question_db_id: int, part_id: Optional[int]) -> list[dict]:
    """
    Fetch steps for a given question and optional part.
    """
    if part_id is None:
        cursor.execute(
            """
            SELECT step_number, description, expression
            FROM steps
            WHERE question_db_id = %s AND part_id IS NULL
            ORDER BY step_number
            """,
            (question_db_id,),
        )
    else:
        cursor.execute(
            """
            SELECT step_number, description, expression
            FROM steps
            WHERE question_db_id = %s AND part_id = %s
            ORDER BY step_number
            """,
            (question_db_id, part_id),
        )

    rows = cursor.fetchall()
    return [
        {
            'step_number': row['step_number'],
            'description': row['description'],
            'expression': row['expression'],
        }
        for row in rows
    ]


def _fetch_image_by_type(cursor, question_db_id: int, image_type: str, part_id: Optional[int] = None) -> dict:
    """
    Fetch one image row by type and format it to API shape.
    """
    if part_id is None:
        cursor.execute(
            """
            SELECT cloudinary_url, description, latex_extracted
            FROM question_images
            WHERE question_db_id = %s AND part_id IS NULL AND image_type = %s
            ORDER BY id
            LIMIT 1
            """,
            (question_db_id, image_type),
        )
    else:
        cursor.execute(
            """
            SELECT cloudinary_url, description, latex_extracted
            FROM question_images
            WHERE question_db_id = %s AND part_id = %s AND image_type = %s
            ORDER BY id
            LIMIT 1
            """,
            (question_db_id, part_id, image_type),
        )

    row = cursor.fetchone()
    if not row:
        return {'present': False, 'description': None, 'cloudinary_url': None, 'latex_extracted': None}

    present = bool(row['cloudinary_url'] or row['description'] or row['latex_extracted'])
    return {
        'present': present,
        'description': row['description'],
        'cloudinary_url': row['cloudinary_url'],
        'latex_extracted': row['latex_extracted'],
    }


def get_questions(filters: dict) -> list[dict]:
    """
    Query stored questions using optional filters.

    Supported filters:
        class, subject, chapter_number, question_type, difficulty_min, difficulty_max,
        confidence_min, confidence_max, question_id, topic_tag, limit, offset.
    """
    filters = filters or {}
    conn = _get_connection()
    if conn is None:
        return []

    try:
        _ensure_schema(conn)
        where_clauses = []
        params: list[Any] = []

        if filters.get('class') is not None:
            where_clauses.append('q.class = %s')
            params.append(_coerce_int(filters.get('class')))
        if filters.get('subject'):
            where_clauses.append('q.subject = %s')
            params.append(filters.get('subject'))
        if filters.get('chapter_number') is not None:
            where_clauses.append('q.chapter_number = %s')
            params.append(_coerce_int(filters.get('chapter_number')))
        if filters.get('question_type'):
            where_clauses.append('q.question_type = %s')
            params.append(filters.get('question_type'))
        if filters.get('difficulty_min') is not None:
            where_clauses.append('q.difficulty_rating >= %s')
            params.append(_coerce_int(filters.get('difficulty_min')))
        if filters.get('difficulty_max') is not None:
            where_clauses.append('q.difficulty_rating <= %s')
            params.append(_coerce_int(filters.get('difficulty_max')))
        if filters.get('confidence_min') is not None:
            where_clauses.append('q.confidence_score >= %s')
            params.append(_coerce_float(filters.get('confidence_min')))
        if filters.get('confidence_max') is not None:
            where_clauses.append('q.confidence_score <= %s')
            params.append(_coerce_float(filters.get('confidence_max')))
        if filters.get('question_id'):
            where_clauses.append('q.question_id = %s')
            params.append(filters.get('question_id'))
        if filters.get('topic_tag'):
            where_clauses.append(
                'EXISTS (SELECT 1 FROM topics_tags t2 WHERE t2.question_db_id = q.id AND t2.tag = %s)'
            )
            params.append(filters.get('topic_tag'))

        where_sql = ''
        if where_clauses:
            where_sql = 'WHERE ' + ' AND '.join(where_clauses)

        limit_value = _coerce_int(filters.get('limit', 200), 200)
        offset_value = _coerce_int(filters.get('offset', 0), 0)

        params.extend([limit_value, offset_value])

        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                f"""
                SELECT
                    q.id,
                    q.question_id,
                    q.class,
                    q.subject,
                    q.chapter_number,
                    q.chapter_name,
                    q.question_type,
                    q.question_text,
                    q.question_latex,
                    q.answer_text,
                    q.answer_latex,
                    q.final_answer,
                    q.difficulty_rating,
                    q.difficulty_reasoning,
                    q.confidence_score,
                    q.created_at,
                    COALESCE(array_agg(tt.tag) FILTER (WHERE tt.tag IS NOT NULL), ARRAY[]::TEXT[]) AS topics_tags
                FROM questions q
                LEFT JOIN topics_tags tt ON tt.question_db_id = q.id
                {where_sql}
                GROUP BY q.id
                ORDER BY q.chapter_number, q.question_id
                LIMIT %s OFFSET %s
                """,
                params,
            )
            question_rows = cursor.fetchall()

            results = []
            for row in question_rows:
                question_db_id = row['id']
                top_question_image = _fetch_image_by_type(cursor, question_db_id, 'question', None)
                top_answer_image = _fetch_image_by_type(cursor, question_db_id, 'answer', None)
                top_steps = _fetch_steps(cursor, question_db_id, None)

                cursor.execute(
                    """
                    SELECT id, part_label, part_text, part_latex, answer_text, answer_latex, final_answer
                    FROM question_parts
                    WHERE question_db_id = %s
                    ORDER BY id
                    """,
                    (question_db_id,),
                )
                part_rows = cursor.fetchall()
                parts = []
                for part in part_rows:
                    part_id = part['id']
                    part_steps = _fetch_steps(cursor, question_db_id, part_id)
                    part_question_image = _fetch_image_by_type(cursor, question_db_id, 'question', part_id)
                    part_answer_image = _fetch_image_by_type(cursor, question_db_id, 'answer', part_id)

                    part_payload = {
                        'part_label': part['part_label'],
                        'part_text': part['part_text'],
                        'part_latex': part['part_latex'],
                        'answer_text': part['answer_text'],
                        'answer_latex': part['answer_latex'],
                        'steps': part_steps,
                        'final_answer': part['final_answer'],
                        'answer_image': part_answer_image,
                    }
                    if part_question_image.get('present'):
                        part_payload['question_image'] = part_question_image
                    parts.append(part_payload)

                results.append(
                    {
                        'question_id': row['question_id'],
                        'class': row['class'],
                        'subject': row['subject'],
                        'chapter_number': row['chapter_number'],
                        'chapter_name': row['chapter_name'],
                        'question_type': row['question_type'],
                        'question_text': row['question_text'],
                        'question_latex': row['question_latex'],
                        'question_image': top_question_image,
                        'parts': parts,
                        'answer_text': row['answer_text'],
                        'answer_latex': row['answer_latex'],
                        'steps': top_steps,
                        'final_answer': row['final_answer'],
                        'answer_image': top_answer_image,
                        'difficulty_rating': row['difficulty_rating'],
                        'difficulty_reasoning': row['difficulty_reasoning'],
                        'topics_tags': [tag for tag in (row['topics_tags'] or []) if tag],
                        'confidence_score': row['confidence_score'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    }
                )

            return results
    except Exception:
        logger.exception('Failed to query questions')
        raise
    finally:
        conn.close()
