from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI

from faq_notes_chat import DEFAULT_ARCHIVE_ROOT, NotesIndex, compact_text

DEFAULT_MODEL = "gpt-5.4"


def extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Model returned empty output.")
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


class ConstellationIdeaMiner:
    def __init__(self, archive_root: Path, model: str = DEFAULT_MODEL) -> None:
        self.archive_root = archive_root.resolve()
        self.model = model
        self.index = NotesIndex(self.archive_root)
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def ensure_api_key(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set.")

    def _collect_note_context(
        self,
        *,
        query: str | None,
        month: str | None,
        max_notes: int,
        max_chars_per_note: int,
    ) -> list[dict[str, Any]]:
        if query:
            candidates = self.index.search(query=query, limit=max_notes, month=month)
            note_ids = [item["note_id"] for item in candidates]
        else:
            candidates = self.index.list_recent(limit=max_notes, month=month)
            note_ids = [item["note_id"] for item in candidates]

        contexts: list[dict[str, Any]] = []
        for note_id in note_ids:
            payload = self.index.read(note_id=note_id, max_chars=max_chars_per_note)
            if not payload.get("found"):
                continue
            contexts.append(
                {
                    "note_id": payload["note_id"],
                    "captured_at": payload.get("captured_at"),
                    "month": payload.get("month"),
                    "source_name": payload.get("source_name"),
                    "transcript_excerpt": payload.get("text", ""),
                }
            )
        return contexts

    def mine_ideas(
        self,
        *,
        query: str | None = None,
        month: str | None = None,
        max_notes: int = 8,
        limit: int = 5,
        max_chars_per_note: int = 2200,
    ) -> dict[str, Any]:
        self.ensure_api_key()
        contexts = self._collect_note_context(
            query=query,
            month=month,
            max_notes=max(1, min(20, max_notes)),
            max_chars_per_note=max(500, min(6000, max_chars_per_note)),
        )
        if not contexts:
            return {
                "ok": True,
                "model": self.model,
                "query": query,
                "month": month,
                "notes_considered": [],
                "ideas": [],
            }

        request = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are Constellation's idea miner. "
                        "Extract the most actionable build ideas from the provided voice notes. "
                        "Favor software projects, automations, integrations, robotics subsystems, data tools, "
                        "and mixed hardware-software prototypes. Merge duplicate ideas from multiple notes. "
                        "Return strict JSON only as an array. Each idea object must contain: "
                        "idea_id, title, summary, why_it_matters, source_note_ids, suggested_repos, "
                        "implementation_hint, next_voice_step, confidence. "
                        "Keep summaries concrete and compact."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "query": query,
                            "month": month,
                            "idea_limit": max(1, min(8, limit)),
                            "notes": contexts,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "max_output_tokens": 1800,
        }
        response = self.client.responses.create(**request)
        parsed = extract_json_payload(response.output_text)
        if not isinstance(parsed, list):
            raise ValueError("Idea miner did not return a JSON array.")
        ideas: list[dict[str, Any]] = []
        for index, item in enumerate(parsed[: max(1, min(8, limit))], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip() or f"Idea {index}"
            idea_id = str(item.get("idea_id", "")).strip() or f"idea_{index}"
            source_note_ids = item.get("source_note_ids", [])
            if not isinstance(source_note_ids, list):
                source_note_ids = []
            suggested_repos = item.get("suggested_repos", [])
            if not isinstance(suggested_repos, list):
                suggested_repos = []
            ideas.append(
                {
                    "idea_id": idea_id,
                    "title": title,
                    "summary": compact_text(str(item.get("summary", "")).strip(), 500),
                    "why_it_matters": compact_text(str(item.get("why_it_matters", "")).strip(), 300),
                    "source_note_ids": [str(note_id) for note_id in source_note_ids[:10]],
                    "suggested_repos": [str(repo) for repo in suggested_repos[:6]],
                    "implementation_hint": compact_text(str(item.get("implementation_hint", "")).strip(), 300),
                    "next_voice_step": compact_text(str(item.get("next_voice_step", "")).strip(), 220),
                    "confidence": item.get("confidence"),
                }
            )
        return {
            "ok": True,
            "model": self.model,
            "query": query,
            "month": month,
            "notes_considered": [
                {
                    "note_id": note["note_id"],
                    "captured_at": note["captured_at"],
                    "month": note["month"],
                }
                for note in contexts
            ],
            "ideas": ideas,
        }

    def build_codex_prompt(
        self,
        *,
        title: str,
        summary: str,
        source_note_ids: list[str],
        repo_name_or_path: str | None = None,
        extra_instruction: str | None = None,
    ) -> dict[str, Any]:
        self.ensure_api_key()
        related_notes = []
        for note_id in source_note_ids[:6]:
            payload = self.index.read(note_id=note_id, max_chars=1800)
            if payload.get("found"):
                related_notes.append(
                    {
                        "note_id": payload["note_id"],
                        "captured_at": payload.get("captured_at"),
                        "text": payload.get("text", ""),
                    }
                )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Write a clean task prompt for Codex CLI. "
                        "The prompt should be practical, implementation-oriented, and concise. "
                        "Include: objective, relevant source context, any explicit repo target, and expected deliverables. "
                        "Return plain text only."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "title": title,
                            "summary": summary,
                            "source_note_ids": source_note_ids,
                            "repo_name_or_path": repo_name_or_path,
                            "extra_instruction": extra_instruction,
                            "related_notes": related_notes,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            max_output_tokens=1400,
        )
        return {
            "ok": True,
            "model": self.model,
            "title": title,
            "source_note_ids": source_note_ids,
            "repo_name_or_path": repo_name_or_path,
            "prompt": response.output_text.strip(),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mine Constellation notes for actionable build ideas.")
    parser.add_argument("--archive-root", default=str(DEFAULT_ARCHIVE_ROOT), help="Notes archive root.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model to use (default: {DEFAULT_MODEL}).")
    subparsers = parser.add_subparsers(dest="command", required=True)

    mine_parser = subparsers.add_parser("mine", help="Extract candidate build ideas from notes.")
    mine_parser.add_argument("--query", help="Optional search query to focus the mining pass.")
    mine_parser.add_argument("--month", help="Optional month filter in YYYY-MM format.")
    mine_parser.add_argument("--max-notes", type=int, default=8)
    mine_parser.add_argument("--limit", type=int, default=5)

    prompt_parser = subparsers.add_parser("prompt", help="Turn a chosen idea into a Codex prompt.")
    prompt_parser.add_argument("--title", required=True)
    prompt_parser.add_argument("--summary", required=True)
    prompt_parser.add_argument("--repo")
    prompt_parser.add_argument("--extra-instruction")
    prompt_parser.add_argument("--source-note-id", action="append", default=[])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    miner = ConstellationIdeaMiner(archive_root=Path(args.archive_root), model=args.model)

    if args.command == "mine":
        payload = miner.mine_ideas(
            query=args.query,
            month=args.month,
            max_notes=args.max_notes,
            limit=args.limit,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0
    if args.command == "prompt":
        payload = miner.build_codex_prompt(
            title=args.title,
            summary=args.summary,
            source_note_ids=[str(note_id) for note_id in args.source_note_id],
            repo_name_or_path=args.repo,
            extra_instruction=args.extra_instruction,
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
