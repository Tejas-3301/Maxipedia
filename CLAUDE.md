# Project: Multimodal Synthetic-Media Forensics & Provenance Reasoning
24-hour hackathon prototype. Source of truth: docs/solution.pdf and docs/architecture.png (read only the parts relevant to the current step).

## Hard rules
- NEVER use an external deepfake-detection API or a fake/real classifier as the decision-maker. Pretrained models (e.g. AASIST) are optional single evidence sources only.
- Only the fusion step produces the verdict. An LLM may only word summaries from structured evidence.
- Final states: likely manipulated / likely authentic / inconclusive. Abstaining is valid.
- Open-source, local, no paid APIs. Never hardcode results for test files.

## Stack
Backend: Python 3.11, FastAPI, SQLite, ffmpeg/ffprobe, exiftool, OpenCV, MediaPipe, librosa/scipy, numpy, imagehash, cryptography (Ed25519).
Frontend: React + Vite + Tailwind.

## Frozen finding schema (additions only, never renames)
{ id, modality: visual|audio|sync|metadata|provenance, check, depth: screen|deep,
  segment:{start,end,unit}, region:[x,y,w,h], score, reliability,
  stance: supports_manipulation|contradicts|neutral|not_assessable,
  rationale, limitations, evidence:{} }
Every check emits this schema. One file per check in backend/checks/.

## Working rules
- Do ONLY the current step, then stop. Do not start the next step.
- Build thin first, with a fallback that still emits valid findings.
- Keep code modular, typed, minimal comments, no unnecessary dependencies.
- At the end of each step: update PROGRESS.md (done / how to run / known issues), then give a 5-line summary.
- Ask before any architecture change.
Use the Python in .venv (3.11/3.12) for all commands