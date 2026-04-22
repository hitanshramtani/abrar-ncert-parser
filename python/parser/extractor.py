import fitz  # PyMuPDF
import re
import logging

logger = logging.getLogger(__name__)


def extract_page_text_sorted(page) -> str:
    """
    Extract text from a page by sorting ALL spans by Y then X position.
    This correctly handles two-column PDF layouts by merging both columns
    into proper reading order.
    """
    blocks = page.get_text('dict')['blocks']
    all_spans = []

    for block in blocks:
        if block.get('type') != 0:  # only text blocks
            continue
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                text = span['text']
                if not text.strip():
                    continue
                all_spans.append({
                    'text': text,
                    'x0':   span['bbox'][0],
                    'y0':   span['bbox'][1],
                    'x1':   span['bbox'][2],
                })

    if not all_spans:
        return ''

    # Group spans that share the same Y row (within Y_TOL pixels)
    Y_TOL = 3
    all_spans.sort(key=lambda s: (round(s['y0'] / Y_TOL), s['x0']))

    lines = []
    used = set()
    i = 0

    while i < len(all_spans):
        if i in used:
            i += 1
            continue

        base_y = all_spans[i]['y0']
        row = [all_spans[i]]
        used.add(i)

        for j in range(i + 1, len(all_spans)):
            if j not in used and abs(all_spans[j]['y0'] - base_y) <= Y_TOL:
                row.append(all_spans[j])
                used.add(j)

        # Sort row by X position (left to right)
        row.sort(key=lambda s: s['x0'])

        # Merge spans: add space only if there's a visible gap
        line_text = ''
        for k, span in enumerate(row):
            if k == 0:
                line_text = span['text']
            else:
                gap = span['x0'] - row[k - 1]['x1']
                line_text += (' ' if gap > 3 else '') + span['text']

        stripped = line_text.strip()
        if stripped:
            lines.append((base_y, stripped))

        i += 1

    lines.sort(key=lambda x: x[0])
    return '\n'.join(t for _, t in lines)


def clean_extracted_text(text: str) -> str:
    """
    Fix artifacts produced by two-column PDF extraction:
    - Lone single/double char lines (subscript artifacts like '1', '2', 'ee', 'ep')
    - Double commas ',,' -> ','
    - Extra whitespace
    NOTE: We intentionally do NOT fix doubled letters here because the
    regex is too aggressive and corrupts real words (e.g. 'small', 'free').
    The question_parser handles any remaining display issues.
    """
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # Skip pure subscript artifact lines: single letter/digit or
        # two chars that look like subscript leftovers e.g. 'ee', 'ep', 'em'
        if re.match(r'^[a-zA-Z]{1,2}$', s) and len(s) <= 2:
            continue
        if re.match(r'^\d{1}$', s):
            continue

        # Fix double commas
        s = re.sub(r',,', ',', s)

        # Normalize whitespace
        s = re.sub(r'[ \t]+', ' ', s).strip()

        if s:
            cleaned.append(s)

    return '\n'.join(cleaned)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Main function: open PDF and extract full clean text with correct
    column-merge ordering.
    """
    doc = fitz.open(pdf_path)
    page_texts = []

    try:
        for page in doc:
            raw = extract_page_text_sorted(page)
            page_texts.append(raw)
    finally:
        doc.close()

    full_text = '\n'.join(page_texts)
    return clean_extracted_text(full_text)


def extract_text_per_page(pdf_path: str) -> list[dict]:
    """
    Extract cleaned text per page.

    Returns:
        [
            {"page_number": 1, "text": "..."},
            {"page_number": 2, "text": "..."},
            ...
        ]
    """
    page_entries: list[dict] = []
    doc = fitz.open(pdf_path)

    try:
        for page_index, page in enumerate(doc, start=1):
            try:
                raw = extract_page_text_sorted(page)
                cleaned = clean_extracted_text(raw)
                page_entries.append(
                    {
                        'page_number': page_index,
                        'text': cleaned,
                    }
                )
            except Exception:
                logger.exception('Failed to extract text from page %s', page_index)
                page_entries.append(
                    {
                        'page_number': page_index,
                        'text': '',
                    }
                )
    finally:
        doc.close()

    return page_entries
