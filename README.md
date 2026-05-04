# Sf Smartlabs:
# NCERT Test Paper Generator (PDF to JSON + Web App)

This project builds chapter-wise NCERT question JSON files from PDFs and provides a web app to generate test papers by class, subject, and chapter.

## What It Does

- Downloads NCERT chapter PDFs (class-wise and subject-wise).
- Parses each PDF into structured question JSON.
- Checks which JSON files are missing.
- Re-processes missing chapters in batch.
- Runs a web UI to generate a test paper from available JSON files.
- Shows `miising` in UI when the selected chapter JSON is not available.

## Project Structure

- `python/downloader.py`: Downloads chapter PDFs into `python/ncert_pdfs/`.
- `python/main.py`: Parses one PDF and writes one chapter JSON.
- `python/check_file_exist.py`: Compares `ncert_pdfs/` vs `output/` and creates `python/missing.txt`.
- `python/process_missing_json.py`: Batch-processes missing JSON entries from `missing.txt`.
- `python/web_test_paper.py`: Flask web app for test paper generation.
- `python/output/`: Generated chapter JSON files.

## Setup

1. Create/activate virtual environment.
2. Install dependencies:

```powershell
cd python
pip install -r requirements.txt
```

## Run Web Test Paper App

```powershell
cd python
python web_test_paper.py
```

Open: `http://127.0.0.1:5000`

### Web Features

- Select class, subject, chapter.
- `Questions Count` input.
- `Use maximum questions` checkbox.
- Generates random test paper from chapter JSON.
- Shows answer for each question.
- If JSON is missing for selected chapter, shows: `miising`.