# Multimodal Synthetic-Media Forensics & Provenance Reasoning

Local, open-source prototype that runs independent forensic checks (visual, audio, audio-video sync, metadata,
provenance) on an uploaded file, normalises every result into one finding schema, and fuses them into an
uncertainty-aware verdict: **likely manipulated / likely authentic / inconclusive**, with the evidence, its
limitations and a signed chain of custody.

- No external detection API and no fake/real classifier decides anything: only the fusion step produces a verdict.
- Abstaining is a valid answer: thin, conflicting, low-coverage or degraded evidence gives *inconclusive*.
- Everything runs locally; no paid services.

## How it works (one paragraph per stage)

1. **Ingest** - the original is stored read-only and hashed (SHA-256); ffmpeg derives 2 fps PNG frames, a 16 kHz
   mono wav, a full ffprobe/exiftool metadata dump and a ≤360p proxy. Every action is appended to a hash-chained,
   Ed25519-signed custody log.
2. **Whole-file checks** (always on the original bytes): metadata & encoding (timestamps, durations, edit lists,
   encoder strings, re-encode generations, C2PA), audio (high-frequency cutoff vs room tone, spectral flatness,
   noise-floor splices), GOP / timestamp gaps, media quality, provenance (SHA-256 ledger, per-segment Merkle tree,
   perceptual-hash lineage graph) and capture-device signatures.
3. **Adaptive deep analysis** - a cheap screening pass draws suspicion curves; the scheduler orders intervals
   (top flagged → random controls → rest) and the deep checks (full-res noise/ELA region maps, frame-level
   temporal discontinuities, MediaPipe lip-sync) run interval by interval until a hard deadline. Coverage, the
   intervals left unanalysed and a control-interval miss-rate are reported.
4. **Fusion** - per-finding log-likelihood ratios (±K·reliability·score), overlap grouping with redundancy discount,
   per-modality posteriors combined by noisy-OR (monotone: clean evidence only lowers P). Clean evidence is scoped
   to the time it actually covered; provenance is capped; degraded media cannot be certified authentic.
5. **Investigator workspace** - video + swimlane timeline, evidence view with heatmaps/spectrograms,
   accept/reject/annotate (logged to custody), evidence graph with a "remove this evidence" counterfactual, attack
   mode, "analyze deeper here", PDF report with the chain of custody.

## Layout
```
backend/
  main.py          FastAPI API            serve.py   API under /api + built UI at / (one process)
  ingest.py runner.py attack.py report.py config.py models.py (frozen finding schema)
  checks/          one file per check: metadata, audio, visual, sync*, provenance, capture, quality (+ registry)
  adaptive/        screening.py, scheduler.py, deep.py (deadline loop)      *sync runs only as a deep unit
  fusion/          fuse.py (verdict), graph.py (evidence graph, counterfactuals)
  provenance/      fingerprint (Merkle), ledger, lineage, graph, custody (Ed25519), device (capture signing)
  db/              SQLite schema + access
frontend/          React + Vite + Tailwind workspace
scripts/           test set, evaluation, results, demo, device simulator
tests/             pytest (62 tests)
data/              runtime data (uploads, proxies, artifacts, forensics.db, keys) - gitignored
*.pdf, *.png       source of truth: "CodeAstra 2.0 ... Build Plan.pdf", "Multimodal Media Forensics Architecture.png"
                   (repo root; docs/ is empty although CLAUDE.md refers to docs/)
```

## Setup

Requirements: **Python 3.11-3.14** (developed on 3.14.7; MediaPipe 1.1.0 ships a py3 wheel), **ffmpeg/ffprobe** on
PATH, **Node 18+** (only to build or develop the UI), optional **exiftool** (richer metadata) and **c2patool**
(C2PA validation).

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\fetch_models.py          # MediaPipe Face Landmarker (Apache-2.0, 3.7 MB) -> models/
.venv\Scripts\python scripts\make_testset.py          # ground-truth test set -> data/testset/ (+ expected.json)
cd frontend; npm install; npm run build; cd ..
.venv\Scripts\python -m pytest -q                     # 62 tests, ~1 min
```
(On macOS/Linux use `.venv/bin/python` and `/` paths.)

## Run

**Single process (recommended for demos)** - API + built UI on one port, no Node at runtime:
```powershell
.venv\Scripts\python scripts\demo_serve.py --data data       # http://127.0.0.1:8000
```
**Development** - API with reload + Vite dev server:
```powershell
.venv\Scripts\python -m uvicorn backend.main:app --reload --port 8000
cd frontend; npm run dev                                       # http://localhost:5173 (proxies /api -> :8000)
```
Vite listens on `localhost` (IPv6 on some Windows machines): use `http://localhost:5173`, not `127.0.0.1`.

Deep links: `#<job>`, `#<job>/<finding>`, `#<job>/<finding>/<graph|findings|custody|evidence>`.

## Demo script (≈7 minutes)

Prepare once on the demo machine (≈2 min): `.venv\Scripts\python scripts\demo_prepare.py`, then serve with
`.venv\Scripts\python scripts\demo_serve.py --open`. `demo_data/DEMO.md` lists every job's link and talking point.

1. **Upload live** (any test file, e.g. `data/testset/03_splice.mp4`): watch the stages light up - the screening
   curve appears first, whole-file findings fill in, then deep findings; the verdict lands in a few seconds.
2. **03_splice - visual edit**: click the red block in the *visual* lane (3.0-4.5 s) → the player jumps and draws
   the box where the moving ball vanished; Evidence tab shows the noise-residual heatmap. Click the thin red mark at
   6.0 s → frame jump; the *provenance* lane says "0.4 s removed at 6.0 s" relative to the trusted original.
3. **02_voice_swap - audio + sync + device**: audio lane red over 2.4-5.6 s (high band missing, noise floor drops
   to digital silence; open the spectrogram), sync lane: lips stop matching; provenance: the file came with the
   camera manifest of 01 → "origin chain broken, audio 2-8 s". Click **Accept** on one finding and **Annotate**
   another - both appear in the Custody tab with signatures.
4. **Graph tab** (on 02 or 03): red/green edges show what pushes the verdict; click a dot to remove that evidence
   → the counterfactual banner shows the verdict changing (✱ marks single findings that flip it). Reject a finding
   in Evidence → "Reviewed verdict" appears; the automatic verdict stays.
5. **05_metadata_edited**: content bit-identical to the original (Merkle root equal) but forged timestamps and device
   tags → manipulated metadata, untouched content.
6. **01 / 06 / 07**: clean original (signed at capture: "origin chain intact"), trimmed/downscaled copy and copy of a
   copy - lineage back to the trusted original, verdict likely authentic.
7. **Attack mode** on 03 → *recompress*: the attacked copy's pixel evidence disappears and the verdict drops to
   **inconclusive** (quality gate) instead of claiming authenticity - the system shows its own limits.
8. **Export PDF report** → verdict, evidence with heatmaps, reviews, and the full chain of custody; the PDF's hash
   is itself appended to the chain.

### Fallback if the live demo fails
- Network/UI issue or slow machine: `scripts\demo_serve.py` (no Vite, no Node, one port) serving the
  **prerecorded** `demo_data/` - every step above works offline because all jobs are precomputed; only new uploads
  need processing.
- Backend crash mid-job: restart - interrupted jobs are resumed automatically (logged as `resumed_after_restart`).
- Nothing runs at all: open `demo_data/reports/*.pdf` (one report per file, with custody) and
  `data/results/results.md` (results table) - static evidence of every claim.
- `demo_data` contains absolute paths: build it on the demo machine; `demo_serve.py` warns if it was moved.

## Results and evaluation

```powershell
.venv\Scripts\python scripts\results_table.py --attacks [--long <dir of long clips>]   # -> data/results/results.md
.venv\Scripts\python scripts\eval_verdicts.py --trust 01_clean.mp4                      # verdict table
.venv\Scripts\python scripts\measure_adaptive.py --testset <dir>                         # deadline vs unlimited timing
```
Latest numbers (see `data/results/results.md`): acceptable verdict on 7/7 test files (6/7 exactly as expected);
screening recall 7/7 ground-truth segments; control miss-rate 0/21 (4% smoothed); time saving −9% on 8 s clips,
+36% on ~55 s 720p clips; attack mode: 9/9 attacked copies of manipulated files become *inconclusive*, none is
called authentic. `expected.json` is used only for scoring - no check or fusion step reads it.

## Capture-device signing (IoT simulation)
```powershell
.venv\Scripts\python scripts\capture_device.py keygen cam01
.venv\Scripts\python scripts\capture_device.py register cam01           # server must be running
.venv\Scripts\python scripts\capture_device.py sign cam01 clip.mp4      # -> clip.mp4.capture.json
curl -F file=@clip.mp4 -F manifest=@clip.mp4.capture.json http://127.0.0.1:8000/api/jobs
```
The `provenance.capture_signature` finding reports *intact* (exact file), *content intact* (re-muxed), *broken*
(changed segments localised via the signed Merkle leaves), *re-encoded* (cannot localise) or *forged* signature.

## API overview (all under `/api` when using `backend.serve`)
| endpoint | purpose |
|---|---|
| `POST /jobs` (file, optional manifest) · `GET /jobs` · `GET /jobs/{id}` | upload, list, status |
| `GET /jobs/{id}/findings` · `/assessment` · `/graph` · `/screening` · `/info` · `/media` · `/artifact?path=` | results, curves, playback, evidence files |
| `POST /jobs/{id}/counterfactual` | verdict without chosen findings / checks / modalities |
| `POST /jobs/{id}/findings/{fid}/review` · `GET /jobs/{id}/reviews` | accept / reject / annotate (custody-logged) |
| `POST /jobs/{id}/deep` | analyze deeper on an interval |
| `POST /jobs/{id}/attack` · `GET /jobs/{id}/attacks` | attack mode and degradation comparison |
| `GET /jobs/{id}/report.pdf` | PDF report with chain of custody |
| `GET /jobs/{id}/custody` · `POST /jobs/{id}/custody` | custody log + verification, investigator note |
| `POST /ledger/{id}/trust` · `GET /provenance/graph` · `POST/GET /devices` | trusted references, lineage, capture devices |

## Limitations (honest list)
- The test set is synthetic (cartoon face, synthetic speech, Windows TTS); thresholds and weights are hand-set,
  not calibrated on a real corpus. Margins on manipulated files are thin (P 0.70-0.77 vs threshold 0.70).
- Lip-sync uses one face; voice-over or multiple speakers read as weak correlation (neutral).
- Region checks assume mostly static framing; camera pans will raise false region anomalies.
- The 0.4 s audio cut in 03 is not found by audio checks (two stretches of identical room tone).
- Short clips gain nothing from adaptive analysis (screening overhead); it pays off on longer, larger files.
- Not implemented: AASIST, speaker embeddings, blockchain anchoring, C2PA validation without c2patool.

## Troubleshooting
- `face_landmarker.task missing` → `scripts\fetch_models.py`.
- Windows console `UnicodeEncodeError` in scripts → `set PYTHONIOENCODING=utf-8` (results script handles it).
- Headless browsers may show a black video frame for H.264; normal Chrome/Edge/Firefox play the proxy.
- Port in use → `scripts\demo_serve.py --port 8001`.
