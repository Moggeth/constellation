# FAQ Notes Tool

USB recorder ingestion and Whisper transcription tool for FAQ voice notes.

## What it does

- Detects connected devices (Windows + Ubuntu/Linux mount points).
- Scans media files from the selected device/path.
- Uses a fast metadata index to skip re-hashing known files, then hashes only unknown candidates.
- Copies new files to monthly folders under this tool folder: `faq_notes/YYYY-MM/audio/`.
- Transcribes with OpenAI Whisper (`whisper-1`) using ffmpeg normalization/chunking.
- Writes monthly transcripts:
  - Combined (time-ordered across all devices): `faq_notes/YYYY-MM/FAQ_YYYY-MM.txt`
  - Per-device file(s): `faq_notes/YYYY-MM/FAQ_YYYY-MM_device_1.txt`, `faq_notes/YYYY-MM/FAQ_YYYY-MM_device_2.txt`

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` on `PATH`
- `OPENAI_API_KEY` environment variable
- Python package: `openai`

Install package:

```bash
pip install openai
```

## Usage

Run from this folder:

```bash
python faq_notes_ingest.py
```

Run from project root:

```bash
python faq_notes_tool/faq_notes_ingest.py
```

Use explicit source path:

```bash
python faq_notes_ingest.py --source "E:\\"
```

Auto-confirm and run without prompt:

```bash
python faq_notes_ingest.py --yes
```

Watch for newly connected devices:

```bash
python faq_notes_ingest.py --watch --yes
```

Set parallel workers (default is parallel already):

```bash
python faq_notes_ingest.py --workers 4
```

Custom archive root:

```bash
python faq_notes_ingest.py --archive-root "./faq_notes"
```

Notes on archive root:
- Default is `faq_notes` resolved relative to the script location, not the current working directory.
- Manifest `copied_path` and `transcript_path` entries are stored archive-relative for portability.

## Output structure

```text
faq_notes_tool/
  faq_notes_ingest.py
  README.md
  faq_notes/
    manifest.json
    YYYY-MM/
      audio/
        YYYYMMDD_HHMMSS_<hash8>.<ext>
      transcripts/
        YYYYMMDD_HHMMSS_<hash8>.txt
      FAQ_YYYY-MM.txt
      FAQ_YYYY-MM_device_1.txt
      FAQ_YYYY-MM_device_2.txt
```
