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
- Optional for PostgreSQL backend: `psycopg[binary]`
- Optional for tray mode on Windows: `pystray`, `Pillow`

Install package:

```bash
pip install openai
```

or install from requirements:

```bash
pip install -r requirements.txt
```

For tray mode:

```bash
pip install pystray pillow
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
python faq_notes_ingest.py --watch
```

Run in Windows system tray (first time each device is seen, approval is required):

```bash
python faq_notes_ingest.py --tray
```

Set parallel workers (default is parallel already):

```bash
python faq_notes_ingest.py --workers 4
```

Use strict duplicate detection (slower, full-file SHA-256):

```bash
python faq_notes_ingest.py --dedupe-mode strict
```

Custom archive root:

```bash
python faq_notes_ingest.py --archive-root "./faq_notes"
```

Use PostgreSQL for ingest state (devices, dedupe index, processed files):

```bash
python faq_notes_ingest.py --storage-backend postgres --account-id "alice" --database-url "postgresql://user:pass@host:5432/faq_notes"
```

Use env vars instead of CLI args:

```bash
set FAQ_NOTES_DATABASE_URL=postgresql://user:pass@host:5432/faq_notes
set FAQ_NOTES_ACCOUNT_ID=alice
python faq_notes_ingest.py --storage-backend postgres
```

One-time migration from local `manifest.json` into PostgreSQL:

```bash
python faq_notes_ingest.py --storage-backend postgres --account-id "alice" --database-url "postgresql://user:pass@host:5432/faq_notes" --migrate-manifest --migrate-only
```

Notes on archive root:
- Default is `faq_notes` resolved relative to the script location, not the current working directory.
- Manifest `copied_path` and `transcript_path` entries are stored archive-relative for portability.

## Watch/tray approval behavior

- In watch/tray mode, newly detected device IDs require manual approval once.
- After approval, that exact device ID is auto-processed on future reconnects.
- If a new device is not approved, it is ignored until approved on a later connection.

## Duplicate detection modes

- `fast` (default): sampled content fingerprint (size + sampled bytes from start/middle/end) for much faster duplicate checks.
- `strict`: full-file SHA-256 (slower, highest confidence).

## Storage backends

- `json` (default): writes `faq_notes/manifest.json` as the ingest index.
- `postgres`: stores ingest index in PostgreSQL tables (`faq_devices`, `faq_files`, `faq_quick_index`).
- Postgres mode requires an account ID (`--account-id` or `FAQ_NOTES_ACCOUNT_ID`) so each user's state is isolated.
- Monthly human-readable outputs are unchanged in both modes:
  - `faq_notes/YYYY-MM/FAQ_YYYY-MM.txt`
  - `faq_notes/YYYY-MM/FAQ_YYYY-MM_device_*.txt`

## Interrupt/disconnect safety

- Source reads now fail gracefully if a USB device disconnects mid-run.
- Audio copies are written to a temporary `.part` file and atomically renamed only when complete.
- Transcript, manifest, and monthly aggregate files are written atomically to avoid partial/corrupted outputs.
- If disconnect happens mid-ingest, already completed items remain intact and the rest can be resumed on reconnect.

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
