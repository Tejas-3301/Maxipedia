# Progress

## Step 1 — Scaffold, schema, DB, health ✅
**Done**
- Repo layout: `backend/{checks,fusion,provenance,db}`, `frontend/` (empty), `data/{uploads,proxies,artifacts}`, `scripts/`, `tests/`.
- `backend/models.py`: `Finding` pydantic model matching the frozen schema (enums for modality/depth/stance; score & reliability bounded 0–1; region `[x,y,w,h]`; segment `{start,end,unit}`).
- `backend/db/store.py`: SQLite tables `jobs`, `findings` (full finding JSON in `data` + indexed columns), `custody_log` (hash-chain fields `prev_hash`/`entry_hash`/`signature` ready for Ed25519).
- `backend/checks/registry.py`: `JobContext`, `CheckFn = (JobContext) -> list[Finding]`, `@register(name, modality, media_types)`, `run_checks()` — a crashing check yields a `not_assessable` finding instead of failing the job.
- `backend/main.py`: FastAPI app, creates data dirs + DB on startup, `GET /health`.

**How to run**
```
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m uvicorn backend.main:app --reload   # GET http://localhost:8000/health
.venv\Scripts\python -m pytest -q                            # 5 tests
```

**Known issues**
- `.venv` is Python **3.14.7**, not 3.11/3.12 as CLAUDE.md specifies. Fine for this step; MediaPipe/OpenCV wheels may not exist for 3.14 — recreate venv with 3.11/3.12 before the visual-checks step.
- No checks registered yet; no upload/job endpoints yet.
- Frontend not initialized (empty dir).
- Source docs live at repo root (`*.pdf`, `*.png`), not `docs/` as CLAUDE.md says.

## Step 2 — Ingestion, job runner, job/findings API ✅
**Done**
- `POST /jobs` (multipart `file`): streams upload to `data/uploads/<job>/original.<ext>`, SHA-256 computed while streaming, file set read-only, job row + `ingested` custody entry (hash-chained, unsigned) → 202.
- `backend/ingest.py` `preprocess()`: ffprobe JSON + exiftool (if installed) → `artifacts/<job>/metadata.json`; media type classify (video/audio/image/other); PNG frames @ 2 fps (lossless, so ELA isn't polluted); 16 kHz mono PCM wav; ≤360p H.264 proxy in `data/proxies/` (PNG proxy for images). Derived artifacts capped at 60 s; original never altered. Each step fails independently into `artifacts.errors`.
- `backend/runner.py`: job thread pool (2) → re-verifies original hash → preprocess → `run_checks` (checks in parallel, 4 workers) → findings to SQLite. Status: `queued → preprocessing → analyzing → done | failed`.
- Registry: auto-discovers every module in `backend/checks/`; `JobContext` now carries typed paths (frames_dir, audio_path, proxy_path, metadata_path, sha256, duration_s, frame_sample_fps).
- Dummy check `ingest_inventory` (neutral, reliability 0, flagged "must be ignored by fusion").
- `GET /jobs/{id}` (status + artifacts + findings_count), `GET /jobs/{id}/findings` (list[Finding]).
- DB: additive columns on `jobs` (size_bytes, duration_s, artifacts, error) via auto-migration; WAL mode; connections now closed per call (`db.session()`).
- `FORENSICS_DATA_DIR` env var overrides data dir (tests use a temp dir).

**How to run**
```
.venv\Scripts\python scripts\make_sample_video.py data\samples\sample.mp4
.venv\Scripts\python -m uvicorn backend.main:app --reload
curl -F "file=@data/samples/sample.mp4" http://localhost:8000/jobs
curl http://localhost:8000/jobs/<job_id>
curl http://localhost:8000/jobs/<job_id>/findings
.venv\Scripts\python -m pytest -q    # 7 tests, incl. end-to-end video ingest
```
Verified live: 4 s sample → 8 PNG frames, 16 kHz mono wav, 360p proxy, metadata.json, read-only original whose hash matches certutil; job `done` in ~0.25 s.

**Known issues**
- exiftool not installed → metadata dump has ffprobe only, `exiftool: {available: false}`. Install (e.g. `winget install OliverBetz.ExifTool`) before the metadata-checks step.
- Six-file test set (PDF step 4) not created yet.
- Job queue is in-process: jobs `queued`/`preprocessing` when the server stops are not resumed on restart.
- Frame timestamps are implied (`t ≈ (i-1)/frame_sample_fps`), not stored per frame.
- Still on Python 3.14 venv (see Step 1).

## Step 3 — Ground-truth test set ✅
**Done**
- `scripts/make_testset.py` → `data/testset/` (7 media files + `expected.json`), deterministic (seed 1337), ~13 s.
  - Base clip: synthetic by default (OpenCV scene: textured bg, per-frame sensor noise, moving ball, cartoon face with mouth driven by the audio envelope; additive-harmonic "speech" + constant pink room tone). `--source clip.mp4` uses a real talking-head clip instead.
  - `01_clean` single encode · `02_voice_swap` TTS in 2.5–5.5 s, video stream bit-exact copy · `03_splice` blurred+JPEG q35 static patch hides moving ball in 3.0–4.6 s at [384,48,179,139] + 0.4 s frames/audio cut at 6.0 s (encoded once from raw frames, no double-compression confound) · `04_reencoded_3x` crf/GOP 23/30 → 27/45 → 24/60 · `05_metadata_edited` stream-copied, forged mvhd time, hex-edited per-track tkhd/mdhd times, fake iPhone make/model, Premiere software tag vs "unedited" comment, fs mtime 2018 · `06_derived_a` trim+downscale of 01 · `07_derived_b` crop+brighten of 06.
  - TTS chain: Piper (`--piper-voice`/`PIPER_VOICE`) → Windows SAPI → espeak-ng → numpy band-limited fallback. This machine used **windows_sapi**.
- `expected.json`: per file ground_truth, expected/acceptable verdicts, parent (lineage), manipulations with segments/regions, should_trigger (modality + signal + segment/region), should_not_trigger, sha256. Evaluation only — checks/fusion must never read it.

**How to run**
```
.venv\Scripts\python scripts\make_testset.py                     # synthetic base
.venv\Scripts\python scripts\make_testset.py --source talk.mp4   # real face (>= 8 s)
```
Verified: video MD5 identical in 01/02/05, audio identical 01/05; 02 swap window HF(>11.5 kHz) energy 0.197 → 6e-7, noise floor −51 dB → digital silence; 03 region noise-residual std 2.33 → 0.65, ball visibly removed; 05 tags show 2019 container / 2024 video / 2023 audio times.

**Known issues**
- 7 files, not 6: "two derived copies" are two files (PDF allows 6–8).
- Synthetic cartoon face likely won't be detected by MediaPipe → sync checks may be not_assessable on synthetic base; use `--source` for real lip-sync testing.
- Signals are deliberately strong (TTS segment has dead-silent floor). A harder variant (attacker adds room tone) may be worth adding for the "attack mode" demo.
- SAPI voice is generic TTS, not a cloned voice; Piper needs a downloaded .onnx voice.
- Filesystem mtime forgery is lost on HTTP upload.

## Step 4 — Metadata & encoding inspection ✅
**Done**
- `backend/checks/metadata.py` (registered as `metadata`, all media types) emits 6 findings per file, each quoting exact field paths/values in `rationale` and `evidence.fields`:
  - `metadata.timestamps` — mvhd vs per-track tkhd/mdhd creation/modification, modification < creation, format/QuickTime/EXIF tags vs mvhd (timezone-like offsets down-weighted), future / pre-1995 dates.
  - `metadata.durations` — audio vs video stream duration (>0.2 s), start_time offset (>0.1 s), container duration overhang.
  - `metadata.edit_list` — multi-segment elst (container-level cut), empty-edit delay, large initial media_time skip, rate ≠ 1.
  - `metadata.encoder` — container/stream tags, EXIF, and H.264/HEVC SEI strings from the bitstream; flags camera make/model vs software encoder, NLE software tags, "original/unedited" claims vs software. Plain FFmpeg/x264 → neutral ("export/transcode").
  - `metadata.reencode_generations` — double-compression hint: P-frames ≥2× local median size away from current I-frames; estimates prior GOP from spike spacing.
  - `provenance.c2pa` — c2patool if installed (validation status, signer, actions); otherwise JUMBF/c2pa byte scan.
- `backend/checks/_mp4.py` — dependency-free MP4/MOV box parser (mvhd/tkhd/mdhd/hdlr/elst, JUMBF); `_`-prefixed so the registry skips it.
- `scripts/run_testset.py` — runs every test file through ingest + all checks in a temp data dir; `--check` filter, `--out` JSON.
- Score convention (used from now on): `score` = strength of evidence for the stated stance; `reliability` = general trustworthiness of that evidence type.

**How to run**
```
.venv\Scripts\python scripts\run_testset.py --check metadata
.venv\Scripts\python -m pytest -q     # 10 tests
```

**Results on test set** (metadata only)
| file | fires (supports_manipulation) | matches expected.json |
|---|---|---|
| 01_clean | none (timestamps/durations contradict, weak) | ✅ |
| 02_voice_swap | none | ✅ (metadata not expected) |
| 03_splice | none | ✅ (should_not_trigger metadata) |
| 04_reencoded_3x | reencode_generations 0.59 — spikes at frames 48/90/136, prior GOP 45 recovered (true gen-2 GOP = 45) | ✅ |
| 05_metadata_edited | timestamps 0.95 (tkhd/mdhd 2024/2023 vs mvhd 2019), encoder 0.92 (iPhone claim vs Lavf+x264, Premiere tag, "unedited" comment) | ✅ |
| 06/07 derived | none; timestamps neutral (stripped) | ✅ |

**Known issues**
- exiftool and c2patool not installed: EXIF-only stills get little coverage; C2PA only byte-detected, never validated.
- Generation hint misses 06_derived_a (one re-encode, single I-frame in 6 s) — absence is reported as neutral, never as authentic.
- Scene cuts can create false P-frame spikes (stated in limitations; reliability 0.4).
- Sub-check names use `metadata.*` / `provenance.*` prefixes inside one registered check; failures of one sub-check yield a not_assessable finding without affecting the others.

## Step 5 — Visual checks (thin) ✅
**Done**
- `backend/checks/visual.py` (registered `visual`, video + image), 4 sub-checks; findings carry `segment` (s), `region` [x,y,w,h], `evidence.frame_range` (native frames) and `evidence.heatmap`:
  - `visual.noise_residual` / `visual.ela` — per-16px-block maps on the 2 fps PNG frames; each frame's global level removed (I/P quality swings), then each block z-scored against **its own temporal median**, so flat-but-genuine areas don't fire. Anomalous blocks → connected components (>= 2% of frame) → tracked across consecutive samples (IoU >= 0.3, >= 2 samples). Single images fall back to spatial-only.
  - `visual.temporal` — native-rate decode at 160x120: DC-free blurred frame differences; isolated spikes (>= 3x both neighbours and local median) = drop/splice, with Farneback optical-flow p99.8 jump ratio; hard cuts; duplicates (only between normally-moving neighbours); container pts gaps.
  - `visual.gop` — I-frame spacing, all-intra, GOP regime change (segments encoded separately). Double-compression stays in `metadata.reencode_generations` (no double counting).
- Heatmaps: `data/artifacts/<job>/heatmaps/{check}_overview.png`, `{check}_event{k}.png` (overlay + box; temporal = prev | next | diff).
- Registry guard: a non-serialisable finding becomes a not_assessable finding instead of failing the job.
- Fix in metadata: exiftool (now installed) `System:`/`File:` tags (our stored copy's fs times) are excluded — they falsely fired timestamps on every file.
- `scripts/run_testset.py`: summary table (`--quiet`), `--keep` writes artifacts to `data/artifacts/`.

**Results (all checks; cell = max supports_manipulation score)**
| file | truth | md.encoder | md.reencode_gen | md.timestamps | visual.ela | visual.noise_residual | visual.temporal | others |
|---|---|---|---|---|---|---|---|---|
| 01_clean | authentic | - | - | - | - | - | - | - |
| 02_voice_swap | manipulated | - | - | - | - | - | - | - (audio/sync not built yet) |
| 03_splice | manipulated | - | - | - | 0.69 | 0.74 region [384,48,176,144] 3.0-4.5 s (truth [384,48,179,139] 3.0-4.6 s) | 0.48 at 6.00 s (truth cut 6.0 s) | - |
| 04_reencoded_3x | recompressed | - | 0.59 | - | - | - | - | - |
| 05_metadata_edited | metadata | 0.92 | - | 0.95 | - | - | - | - |
| 06/07 derived | derivative | - | - | - | - | - | - | - |
No false positives on the test set.

**How to run**
```
.venv\Scripts\python scripts\run_testset.py --keep --quiet     # table; heatmaps in data/artifacts/<job>/heatmaps
.venv\Scripts\python -m pytest -q                               # 13 tests
```

**Known issues**
- Drop detector margin is thin on the test set (splice isolation 3.36 vs 2.83 max elsewhere); reliability kept at 0.45.
- Region checks use 2 fps samples: edits shorter than ~1 s (2 samples) are not caught; camera motion/pans will raise false region anomalies on real footage (temporal baseline assumes a mostly static framing).
- ELA on decoded video is weak by nature (reliability 0.35).
- Duplicates in low-motion stretches are ignored (indistinguishable from encoder P-skips).
- Face detection / face-vs-background blending skipped (per instructions).
- Runs on the whole clip (<= 60 s), not yet on scheduler-flagged intervals.

## Step 6 — Audio checks (thin) ✅
**Done**
- `backend/checks/audio.py` (registered `audio`, video + audio). Decodes the original's first audio stream at **native rate** (the 16 kHz wav can't show cutoffs above 8 kHz), peak-normalised; numpy STFT (no scipy/librosa needed). All judgements are relative to the clip's own baseline.
  - `audio.hf_cutoff` — reference = high-band level of **pause frames** (room tone, digital silence excluded). Room tone is additive, so genuine speech never drops below it; active windows > 20 dB below it are merged (gap <= 0.75 s) into segments. Reports band edge (vs room-tone spectrum), clip cutoff, deficit, segment vs clip spectral flatness. Whole-clip band-limit (upsampled source) as a weak fallback.
  - `audio.spectral_flatness` — per-window median flatness of active frames vs clip baseline (robust z); only long (>= 1 s), large (x2.5) departures; reliability 0.25.
  - `audio.noise_floor` — 1 s windows (long enough to contain pauses), 5th-percentile 20 ms energy; steps > 8 dB or digital-silence dropouts inside room tone → segment with refined splice points.
- Spectrograms: `data/artifacts/<job>/spectrograms/overview.png` + `{check}_event{k}.png` crops (segment ±0.75 s, white boundary markers, kHz grid).
- Bugs fixed while building: frame-energy floor now ignores digital silence (gaps dragged it to −200 dB and disabled the HF check).

**Results (all checks)**
| file | truth | audio.hf_cutoff | audio.noise_floor | visual | metadata |
|---|---|---|---|---|---|
| 01_clean | authentic | - | - | - | - |
| 02_voice_swap | manipulated | **0.77** 2.44–5.62 s, band edge ~5.6 kHz, −41 dB vs room tone | **0.75** 3.74–5.18 s digital silence | - | - |
| 03_splice | manipulated | - | - | ELA 0.69, noise 0.74, temporal 0.48 | - |
| 04_reencoded_3x | recompressed | - | - | - | reencode 0.59 |
| 05_metadata_edited | metadata | - | - | - | encoder 0.92, timestamps 0.95 |
| 06/07 derived | derivative | - | - | - | - |
Truth for 02: TTS in 2.5–5.5 s. Spectral flatness fires on nothing (expected for a thin indicator). No false positives.

**How to run**
```
.venv\Scripts\python scripts\run_testset.py --keep --quiet
.venv\Scripts\python -m pytest -q      # 15 tests
```

**Known issues**
- 03's 0.4 s audio cut is **not** detected: it joins two stretches of the same continuous room tone; needs phase/ENF continuity or a splice-click model.
- HF check needs room tone in pauses; on studio/noise-gated audio it falls back to the whole-clip band-limit rule only.
- Noise-floor check only catches the digital-silence part of the 02 swap (the TTS speech itself has no pauses, so its floor isn't measurable within 1 s windows).
- Flatness is content-dependent; low reliability by design. Speaker-embedding comparison and AASIST not implemented.

## Step 7 — Fusion, assessment endpoint, minimal frontend ✅
**Done**
- `backend/fusion/fuse.py` — the only place a verdict is made:
  - llr per finding = ±K·reliability·score (K=5; supports +, contradicts −; neutral/not_assessable 0).
  - Grouping: union-find on time overlap (±0.25 s) and region overlap when both have regions; file-level findings share one group. Same-modality findings in a group are redundant: strongest ×1, rest ×0.5.
  - Per-modality posterior σ(logit(0.10) + Σllr); file P(manipulated) = noisy-OR over assessed modalities (absence of visual edits can't cancel an audio splice).
  - States: likely manipulated P ≥ 0.70; likely authentic P ≤ 0.20 **and** ≥ 2 modalities assessed; else inconclusive. Forced inconclusive: < 3 assessable findings ("insufficient evidence"), or strong supports+contradicts (≥ 1.5 each) within the deciding modality ("conflict").
  - Output: state, P, confidence (null when abstaining), reasons, template summary (no LLM), per-modality llr/posterior, top supporting/contradicting findings with effective llr, groups, limitations (unassessed modalities, failed checks, top-evidence limitations, uncalibrated weights). `fuse(findings, exclude=ids)` supported for a later counterfactual toggle.
- Runner stores `verdict`/`confidence` on the job and logs them to custody.
- API: `GET /jobs/{id}/assessment` (409 until done), `GET /jobs` (recent).
- Frontend (`frontend/`, Vite 6 + React 18 + Tailwind 4): upload, recent jobs, status polling, assessment card, findings table with expandable rationale/limitations/evidence JSON; `#<job_id>` deep link; `/api` proxied to :8000.
- `scripts/run_api_flow.py` — HTTP end-to-end over every test file (use `--base http://localhost:5173/api` to go through the Vite proxy).

**Full flow on the test set (via Vite proxy)**
| file | verdict | P(manip) | conf | expected |
|---|---|---|---|---|
| 01_clean | likely authentic | 0.17 | 0.83 | likely authentic ✅ |
| 02_voice_swap | likely manipulated | 0.76 | 0.76 | likely manipulated ✅ (audio.hf_cutoff, audio.noise_floor) |
| 03_splice | likely manipulated | 0.84 | 0.84 | likely manipulated ✅ (visual.noise_residual, temporal, ela) |
| 04_reencoded_3x | inconclusive | 0.28 | – | inconclusive ✅ |
| 05_metadata_edited | likely manipulated | 0.86 | 0.86 | likely manipulated ✅ (timestamps, encoder) |
| 06_derived_a | likely authentic | 0.18 | 0.82 | likely authentic ✅ |
| 07_derived_b | likely authentic | 0.18 | 0.82 | likely authentic ✅ |
~2–3 s per file. UI verified with a headless Edge screenshot of the 02 job.

**How to run**
```
.venv\Scripts\python -m uvicorn backend.main:app --port 8000
cd frontend && npm install && npm run dev            # http://localhost:5173
.venv\Scripts\python scripts\run_api_flow.py --base http://localhost:5173/api
.venv\Scripts\python -m pytest -q                     # 22 tests
```

**Known issues**
- Margins are thin: 06/07 at P=0.18 vs authentic threshold 0.20; 02 at 0.76 vs 0.70. K, prior and thresholds are hand-set constants (stated in limitations), not calibrated.
- Sync and provenance (lineage/hash) modalities not built, so they are always listed as "not assessed".
- No evidence graph / counterfactual UI yet (fusion already accepts `exclude`).
- Frontend shows no heatmaps/spectrograms (artifacts are not served over HTTP yet).
- Vite binds to `localhost` (IPv6 on this machine); use `localhost`, not `127.0.0.1`, for :5173.

## Step 8 — Provenance ✅
**Done** (`backend/provenance/`, check `backend/checks/provenance.py`, modality `provenance`)
- `fingerprint.py` — per-packet SHA-256 via `ffmpeg -c copy -f framehash` (no decode, independent of container metadata); leaves = 1 s segment per stream; Merkle root + inclusion proofs; 10 fps thumbnails + 64-bit DCT pHash per frame.
- `ledger.py` — `versions` table: whole-file SHA-256, Merkle root/leaves, packet set, pHashes, thumbnails (`provenance_thumbs.npz`), size/duration, `trusted` flag + label.
- `lineage.py` — near-duplicate by pHash (<= 16/64 bits, >= 60% frames; unrelated ≈ 28-32 bits); time alignment by DP over offsets on normalised thumbnails with crop search (1.0/0.9/0.8) → trim start, coverage, crop, **cuts/insertions** (offset jumps).
- `graph.py` — `lineage_edges` table (relations stored both ways); parent = closest plausible ancestor: consistent timeline > similarity > nearest length/size > trusted.
- `custody.py` — per-job hash chain, entry_hash = SHA-256(job, action, actor, detail, ts, prev); Ed25519 signature over entry_hash; key in `data/keys/` (gitignored). `verify()` detects edits, deletions, reordering, bad signatures; legacy unsigned rows reported as such. Replaces the old unsigned `db.log_custody`.
- Findings: `provenance.sha256_ledger` (exact match), `provenance.merkle` (changed segments per stream vs best packet-sharing version + proofs), `provenance.lineage` (parent, ops, cuts, component graph). Only **trusted** references yield supports/contradicts; untrusted relatives are neutral (direction unknowable).
- API: `POST /ledger/{job}/trust`, `GET /provenance/graph?job_id=`, `GET /jobs/{id}/custody` (entries + verification + public key), `POST /jobs/{id}/custody` (investigator annotation, signed).
- `scripts/run_api_flow.py --trust 01_clean.mp4 --provenance`.

**Results (fresh data dir, 01 registered as trusted original)**
| file | provenance findings |
|---|---|
| 02_voice_swap | merkle **supports 0.6**: video 8/8 segments bit-identical, audio changed 2-8 s; lineage: same timeline as 01 |
| 03_splice | lineage **supports 0.7**: "0.4 s removed at 6.0 s"; merkle supports 0.45 (re-rendered) |
| 04_reencoded_3x | lineage contradicts 0.6 (same timeline, re-encoded) |
| 05_metadata_edited | merkle **contradicts 0.8**: Merkle root identical to 01 → content unaltered, only container differs |
| 06_derived_a | lineage → 01: covers 1.0-7.0 s, 640→480 px, no cuts (contradicts 0.6) |
| 07_derived_b | lineage → 06: central 90% crop, 480→432 px, no cuts (neutral: 06 is not trusted) |
Graph: 01 → {02, 03, 04, 05, 06}, 06 → 07. All 7 custody chains verify intact. Verdicts unchanged and all OK (02 0.78, 03 0.92, 05 0.86, 01/06/07 authentic, 04 inconclusive).

**Known issues**
- Fusion quirk: a provenance *contradiction* still nudges P(manipulated) up slightly (06: 0.183 → 0.19), because noisy-OR over assessed modalities adds each newly assessed modality's residual prior. Needs a fusion tweak (ask first).
- Trust does not propagate: 07's parent 06 is a clean derivative of the trusted 01, but 07 only gets a neutral finding.
- Merkle "changed" means re-encoded *or* edited; for fully re-encoded copies (03, 04, 06, 07) only lineage is informative.
- Lineage alignment is O(n·m) per pair and compares against every ledger entry with a pHash pre-filter; fine for a demo ledger, not for thousands of versions.
- Older jobs in the real `data/` DB have legacy unsigned custody entries (reported as such by verify).

## Step 9 — Adaptive analysis (screening → scheduler → deep) ✅
**Done**
- `backend/adaptive/screening.py` — suspicion curves in 0.25 s bins, per modality + combined:
  video `frame_diff` (proxy, native rate, isolation spikes), `noise_region` (2 fps ingest frames ≤ 640 px — the crf28 proxy erases sensor noise, measured); audio `hf_drop` (vs room tone), `noise_floor` jumps, `spectral_flux` (**display only**: fires on ~40% of any speech clip, measured). Face-present flag reports "unavailable" (no Haar model in this OpenCV build, no MediaPipe on py3.14). Screening findings are `depth=screen`, neutral (timeline only, never evidence).
- `backend/adaptive/scheduler.py` — peaks ≥ 0.35 → stretch ≥ half-peak, ±0.75 s pad, merge (gap ≤ 0.5 s), added in suspicion order while the estimated deep cost fits 75% of `DEEP_BUDGET_S` (budget sets K); 2 random **control** intervals (1.5 s, seeded by file hash) in unflagged time; coverage = union / duration. Cost model `DEEP_COST_S_PER_MPX_S = 0.6` (measured 0.5–0.8).
- Deep visual checks restricted to intervals: noise/ELA on full-res frames at 4 fps inside intervals, z-scored against a clip-wide baseline of ingest frames; region anomalies must persist ≥ 0.75 s; temporal (dup/drop/flow) at native rate per interval ±0.5 s. GOP, pts gaps, metadata, provenance, audio still run on the whole original.
- Schema: `Finding.coverage` (added, optional); deep findings carry `depth=deep`, `coverage`, `evidence.interval`.
- Fusion: assessment `coverage` block (deep coverage, intervals, controls, control hits) + limitation sentence when < 100%. (Fusion maths unchanged.)
- Runner: preprocess → screening → schedule → checks → fuse; status `screening` added; artifacts store curves + schedule.
- API: `GET /jobs/{id}/screening` (curves + schedule), `POST /jobs/{id}/deep {start,end,actor}` ("Analyze deeper here": adds a manual interval, re-runs deep visual checks, replaces `visual.*` findings, re-fuses, signed custody entry).
- `scripts/measure_adaptive.py` — real timings full vs adaptive, coverage, truth-in-flagged, control hits, both verdicts.
- Fixes: ingest now creates the proxies dir itself (scripts had silently produced no proxy); `_probe_video`/`_packets` cached.

**Measured (test set, 8 s 640×480)**
| file | coverage | flagged | truth in flagged | controls hit | verdict full = adaptive |
|---|---|---|---|---|---|
| 01_clean | 38% (controls only) | – | n/a | 0/2 | likely authentic |
| 02_voice_swap | 56% | 1.8–6.2 s | 3/3 | 0/0 | likely manipulated |
| 03_splice | 82% | 2.2–7.0 s | 4/4 | 0/1 | likely manipulated |
| 04_reencoded_3x | 38% | – | n/a | 0/2 | inconclusive |
| 05_metadata_edited | 38% | – | n/a | 0/2 | likely manipulated |
| 06_derived_a | 50% | – | n/a | 0/2 | likely authentic |
| 07_derived_b | 46% | 0.0–1.2 s | n/a | 0/1 | likely authentic |
Time: **no saving on 8 s clips** (full 12.4 s vs adaptive 13.4 s incl. ~0.45 s screening per file).
**Long clips (≈55 s, 1280×720, looped 01/03):** full 62.2 s vs adaptive 50.8 s incl. screening → **18% saving** (21% on an earlier run); coverage 28% / 53%; loop seams correctly flagged; controls 1/4 hit (spliced loop had more edits than the budget allowed — the control estimate exposes that).

**How to run**
```
.venv\Scripts\python scripts\measure_adaptive.py [--testset DIR] [--budget 20]
curl http://localhost:8000/jobs/<id>/screening
curl -X POST http://localhost:8000/jobs/<id>/deep -H "Content-Type: application/json" -d "{\"start\":5,\"end\":6,\"actor\":\"alice\"}"
.venv\Scripts\python -m pytest -q      # 32 tests
```

**Known issues**
- Cost model undershoots on clips with many events (03 long: 30 s actual vs 20 s budget) — events add heatmaps/flow work; budget is an estimate, not a hard stop.
- Short clips gain nothing (fixed overhead); adaptive pays off from roughly ≥ 30 s / ≥ 720p.
- Control intervals are re-drawn when a manual interval is added (they must stay outside flagged/manual time).
- Fusion still treats partial-coverage contradictions like full-coverage ones; scaling `contradicts` by coverage is the natural next step (fusion change → ask first). The earlier noisy-OR monotonicity question is also still open.
- `measured_checks_s` in the schedule times all checks (metadata/audio/provenance too), not only deep visual.
- No frontend for curves / "Analyze deeper here" yet (API only).

## Step 10 — Audio-video sync ✅
**Done**
- `backend/checks/sync.py` (check `sync`, video): MediaPipe **Face Landmarker** (Tasks API; mediapipe 1.1.0 has no legacy `solutions.face_mesh`) → inner-lip gap (landmarks 13/14) / inter-ocular distance (33/263) at 25 fps; speech-band (300–3400 Hz) log energy from the 16 kHz wav at the same rate; 2 s windows, 0.5 s hop, normalised cross-correlation over ±0.4 s lags.
  - clip offset = median best lag of confident windows (r ≥ 0.3); flagged: correlation at clip offset < 0.1 (**drop**) or best lag > 80 ms from the clip offset with r ≥ 0.4 (**drift**); constant |offset| > 200 ms reported weakly.
  - windows with < 70% face frames → `no_face`, < 30% speech frames → `no_speech` → merged `not_assessable` findings.
  - offset curve: `artifacts/<job>/sync_curve.json` (per window: start/end, face/speech fractions, best lag, r_max, r at clip offset, flags) + `heatmaps/sync_offset_curve.png`.
  - clips ≤ 20 s analysed whole; longer clips use the scheduler's deep intervals (≥ 4 s spans); `depth=deep`, `coverage` set.
- `scripts/fetch_models.py` → `models/face_landmarker.task` (Apache-2.0, 3.7 MB, sha256 64184e22…; gitignored). Requirements: mediapipe, opencv-contrib-python (mediapipe replaced opencv-python with the contrib build, same 5.0.0).

**Results**
| file | sync |
|---|---|
| 01, 04, 05, 06, 07 | contradicts: lips match speech in all windows, median r 0.83–0.91 at 0 ms |
| 02_voice_swap | **supports 0.84**, 2.40–5.84 s (truth 2.5–5.5): r falls to 0.04 vs clip 0.63, best lag jumps to −240/−280 ms |
| 03_splice | contradicts (cut removed audio and video together, sync preserved — correct) |
The synthetic cartoon face is detected in 40/40 sampled frames.

**How to run**
```
.venv\Scripts\python scripts\fetch_models.py          # once
.venv\Scripts\python scripts\run_testset.py --check sync
.venv\Scripts\python -m pytest -q                     # 37 tests
```

**Known issues**
- **Fusion monotonicity bug now changes verdicts**: adding sync (contradicts on clean files) *raises* P(manipulated): 01 0.171→0.195, 06/07 0.183→0.206 → **inconclusive** instead of likely authentic (still "acceptable" in expected.json). Cause: noisy-OR over assessed modalities adds each new modality's residual prior. Needs a fusion change — awaiting approval.
- Synthetic mouth is driven by the same audio envelope, so r values on the test set (0.6–0.9) are higher than real speech will give (~0.3–0.6); thresholds may need tuning on real footage.
- One face only (largest/first); multiple speakers or off-screen voice-over will read as weak correlation (reported neutral, not as manipulation).
- Face Landmarker runs ~25 fps on CPU; 60 s clips cost several seconds — hence interval restriction above 20 s.

## Step 11 — Fixes: hard deadline, coverage-aware + monotone fusion, provenance weighting ✅
**1. Scheduler → hard deadline** (supersedes the Step 9 cost model; `DEEP_COST_S_PER_MPX_S` / `ADAPTIVE` removed)
- `scheduler.schedule()` now only orders intervals: manual → top flagged → 2 random controls → remaining flagged (desc. suspicion) → remaining unflagged 2 s chunks (seeded shuffle). Every analysed unflagged chunk is a control.
- `backend/adaptive/deep.py` `run_deep()`: checks elapsed time **before** each interval, stops at `DEEP_BUDGET_S` (20 s); per interval runs the deep units in parallel: `visual.region` (noise + ELA, one decode), `visual.temporal`, **`sync`**. Reports `elapsed_s`, `analysed` (with started/took), `unanalysed`, coverage, control hits, Laplace miss-rate `(hits+1)/(controls+2)`.
- Sync is no longer a registered check: it runs only inside the deadline loop or via "analyze deeper" (manual interval first; `replace_findings` now takes exact check names). The registered `visual` check keeps only whole-file work (`visual.pts_gaps`, `visual.gop`; spatial path for images).
- Deep-unit fixes found while doing this: temporal events only where the ±8-frame neighbourhood lies inside the decoded span (seek + one-sided medians caused false jumps at interval edges); region maps ignore blocks whose residual is < 0.8 grey RMS in both frame and baseline (x24 "ratios" of near-zero noise in flat areas); sync spans widened by ½ window and snapped to an absolute hop grid so boundary-straddling windows are evaluated and de-duplicated.

**Long spliced clip (53 s 720p, 7 splices), budget 20 s**: elapsed **21.36 s** (overshoot = the interval already running), unlimited run 49.75 s; coverage 52%; **unanalysed: 32.8–39.0, 40.2–46.8, 47.8–52.8 s (flagged)** + 6 controls. Long clean loop: 22.21 s vs 20 s (unlimited 74.11 s), coverage 30%, 19 controls unanalysed.

**2. Fusion, partial coverage** — deep clean findings are emitted per analysed interval (segment = interval, `coverage` = its share); llr × coverage, per-check total capped at 1 − miss-rate. supports never discounted. Sync: not-assessable windows produce no clean evidence and no coverage; clean sync findings span only windows with a usable, stable offset. "Likely authentic" additionally needs deep coverage ≥ 60% and no unanalysed flagged interval, unless one contradicting finding is strong (effective llr ≤ −2); otherwise inconclusive with the reason in `reasons` and limitations.

**3. Fusion, monotone + provenance** — every applicable modality starts at its prior (file prior 0.25 split across modalities; video = 5, audio = 3, image = 3), so contradicts can only lower P, supports only raise it, neutral/not_assessable move nothing (fixes the Step 8/10 bug). Provenance: untrusted SHA/Merkle-root match → weak contradicts (0.3 × 0.4); trusted → strong; Merkle mismatch supports only when the tree localizes it (some leaves identical, some changed), otherwise neutral; any single provenance finding can lower the log-odds by at most 3.0. ("Trusted" = explicitly marked; device signing is not implemented.)

**4.** `test_preprocess_from_script_creates_proxy_dir` (proxies dir missing → ingest still produces the proxy).
**5.** Sync schema sanity: tests validate every sync finding against `Finding` and assert `depth == "deep"` (incl. not-assessable ones); not-assessable findings carry `coverage=None`.

**Verdicts, test set (01 registered trusted; same procedure before/after: `scripts/eval_verdicts.py --trust 01_clean.mp4`)**
| file | expected | before | P before | after | P after | deep coverage |
|---|---|---|---|---|---|---|
| 01_clean | likely authentic | likely authentic | 0.195 | likely authentic | 0.169 | 100% |
| 02_voice_swap | likely manipulated | likely manipulated | 0.886 | likely manipulated | **0.701** | 100% |
| 03_splice | likely manipulated | likely manipulated | 0.872 | likely manipulated | 0.727 | 100% |
| 04_reencoded_3x | inconclusive | inconclusive | 0.312 | **likely authentic** | 0.195 | 100% |
| 05_metadata_edited | likely manipulated | likely manipulated | 0.866 | likely manipulated | 0.765 | 100% |
| 06_derived_a | likely authentic | inconclusive | 0.216 | **likely authentic** | 0.134 | 100% |
| 07_derived_b | likely authentic | inconclusive | 0.206 | **likely authentic** | 0.178 | 100% |
All 7 within acceptable verdicts. Short clips now reach 100% deep coverage because unflagged chunks fill the remaining deadline.

**How to run**
```
.venv\Scripts\python scripts\eval_verdicts.py --trust 01_clean.mp4 [--compare before.json]
.venv\Scripts\python scripts\measure_adaptive.py --testset <dir> [--budget 20]
.venv\Scripts\python -m pytest -q     # 47 tests
```

**Known issues**
- Manipulated margins shrank: 02 at P=0.701 vs T_MANIP 0.70 (two more unassessed-at-prior modalities and capped provenance now share the probability mass). 04 now "likely authentic" (trusted-lineage contradiction is strong) — acceptable per expected.json but no longer the expected "inconclusive".
- The deadline is checked between intervals; one long interval can overshoot (measured ≤ 2.2 s at 720p).
- File prior 0.25, thresholds and weights remain hand-set.
- Long looped clean clip is "likely manipulated" — correct: the loop seams are real cuts.

## Step 12 — Evidence graph + "remove this evidence" counterfactual ✅
**Done**
- `backend/fusion/graph.py`, built from fusion's own intermediate values (so it explains the verdict exactly) and returned as `assessment.graph`:
  - nodes: `verdict` (state, P), `mod:<modality>` (posterior, llr, assessed, conflict), findings (stance, depth, coverage, segment, region, effective llr, group, role = evidence | context), `group:<id>` (overlap clusters).
  - edges: finding → modality and modality → verdict, `supports` / `contradicts`, weighted by effective llr; finding ↔ finding inside a group: `redundant_with` (same modality, same stance — the discounted ones), `conflicts` (same modality, opposite stance), `corroborates` (same stance, different modalities, localized in time/region). Opposite stances across modalities get no edge (independent channels, as in fusion). neutral / not_assessable findings are context nodes without weighted edges.
- Counterfactual: `counterfactual(findings, exclude, ...)` and `impacts()` (P and verdict with each weighted finding removed).
- API: `GET /jobs/{id}/graph` (graph + per-finding `impact`: p_without, state_without, delta_p, flips_verdict); `POST /jobs/{id}/counterfactual {exclude: [ids], exclude_checks, exclude_modalities, actor}` → before/after/delta/state_changed + full alternative assessment. Stored verdict is never changed; with an `actor` the request is logged (signed) to custody. Unknown ids / empty request → 400.
- Frontend: findings table gets a "remove" checkbox on weighted findings and an "If removed" column (P without it, ✱ = verdict flips); ticking calls the counterfactual endpoint and shows a dashed counterfactual banner with reset.

**Results (live API, 01 trusted)**
| file | single removal that flips | remove modality | remove all supporting |
|---|---|---|---|
| 02_voice_swap (P 0.70) | audio.hf_cutoff → 0.51, audio.noise_floor → 0.52, sync.lipsync → 0.64 (inconclusive) | audio → 0.33 inconclusive | likely authentic 0.16 |
| 03_splice (P 0.73) | provenance.lineage → 0.56, visual.noise_residual → 0.61, provenance.merkle → 0.62 | visual → 0.58 inconclusive | likely authentic 0.18 |
| 05_metadata_edited (P 0.77) | metadata.timestamps → 0.43, metadata.encoder → 0.50 | metadata → 0.14 likely authentic | likely authentic 0.13 |
UI verified with a headless Edge screenshot of the 02 job (impact column + ✱ markers).

**How to run**
```
curl http://localhost:8000/jobs/<id>/graph
curl -X POST http://localhost:8000/jobs/<id>/counterfactual -H "Content-Type: application/json" -d "{\"exclude_modalities\":[\"audio\"],\"actor\":\"alice\"}"
.venv\Scripts\python -m pytest -q      # 51 tests
```

**Known issues**
- Every 02/03/05 verdict hinges on a single finding (thin margins over T_MANIP = 0.70) — the counterfactual makes this visible rather than fixing it.
- The UI sends `actor: "investigator"`, so each toggle adds a custody entry (honest, but chatty).
- No graph drawing in the UI yet (API returns nodes/edges; frontend shows only the impact column + counterfactual banner).
- Impacts recompute fusion once per weighted finding (~30 per job): instant now, would need caching for very large finding sets.

## Step 13 — Investigator workspace ✅
**Done**
- Backend:
  - Staged pipeline so the UI fills in progressively: `preprocessing → screening` (curves + ordered intervals saved) `→ analyzing` (whole-file findings + screening findings saved) `→ deep` (deep findings saved) `→ done` (verdict).
  - `GET /jobs/{id}/media` (proxy, HTTP Range → seeking), `GET /jobs/{id}/info` (original width/height/fps: region boxes are in original pixels), `GET /jobs/{id}/artifact?path=` (heatmaps/spectrograms/sync curve; resolved path must lie inside `artifacts/<job>/`, else 404).
  - `POST /jobs/{id}/findings/{fid}/review {decision: accept|reject|note|clear, note, actor}` → signed custody entry; `GET /jobs/{id}/reviews` rebuilds review state from the custody chain (single source of truth). The stored verdict is not changed.
- Frontend (`frontend/src/components/`):
  - **Player**: proxy video; selected finding's region box drawn in original-pixel coordinates (scaled), optional heatmap overlay, "Analyze deeper here (t ± 1.5 s)".
  - **Timeline**: swimlanes visual / audio / sync / metadata / provenance; screening curve per lane (grey area + flag level) appears first, deep findings fill in (red = supports, height/opacity ∝ score; green = clean, scoped to its interval; hatched = not assessable); sync lane shows the lip-sync correlation curve; top row = deep schedule (analysed solid, unanalysed outlined/hatched); file-level findings as chips; playhead; click finding → jump + box; click lane → seek. Reviewed findings: accepted = blue outline, rejected = dashed + faded.
  - **Evidence** tab: rationale, limitations, images from evidence, metadata `fields` table, raw evidence; Accept / Reject / Clear / Annotate (investigator name from header, kept in localStorage) with history and entry hashes.
  - **Graph** tab: findings → modalities → verdict (supports red / contradicts green, width ∝ weight), corroborates / conflicts / redundant arcs, impact Δ per finding (✱ = flips); click dot = remove/restore (counterfactual), click label = open evidence; "remove my rejected" button; verdict node shows counterfactual result.
  - **Findings** tab (table with remove toggles, review state, impact) and **Custody** tab (entries + chain verification).
  - "Reviewed verdict" banner: counterfactual without the evidence the investigator rejected (automatic verdict untouched).
  - Deep link `#<job>/<finding>/<tab>`.
- Exploratory counterfactual toggles are no longer logged to custody (reviews, deep requests and trust changes are).

**Verified**: headless Edge screenshots of the 03 job — timeline with region event 3.0–4.5 s and cut at 6.0 s, box at [384,48,176,144] drawn on the player at t = 3.05 s, evidence view with heatmap and alice/bob review history from the custody chain, graph view with ✱ impacts.

**How to run**
```
.venv\Scripts\python -m uvicorn backend.main:app --port 8000
cd frontend && npm run dev        # http://localhost:5173   (link: #<job>/<finding>/<tab>)
.venv\Scripts\python -m pytest -q # 55 tests
```

**Known issues**
- Headless Edge did not paint H.264 frames in screenshots (black player); overlays, seeking and controls were verified, real playback should be checked in a normal browser.
- Audio has no waveform/spectrogram strip on the timeline yet (spectrogram crops are in the evidence view).
- PDF report export (PDF step 5) not done.
- Graph layout is a fixed 3-column SVG; very large finding sets will get long.

## Step 14 — PDF report, attack mode, results table ✅
**Done**
- **PDF report** (`backend/report.py`, fpdf2; `GET /jobs/{id}/report.pdf?actor=`): file identity + SHA-256, fusion verdict/P/confidence/reasons, coverage + miss-rate, per-modality table, reviewed verdict (rejections removed), limitations, key evidence with rationale/limitations/annotations and heatmap/spectrogram/sync-plot images + metadata field tables, all findings with review state, provenance rationales, full chain of custody (entries, verification, key id, head, public key PEM). The PDF's SHA-256 is appended as a signed `report_exported` custody entry (also in `X-Report-SHA256`). Text is normalised to Latin-1 (no platform fonts).
- **Attack mode** (`backend/attack.py`; `POST /jobs/{id}/attack {kind: recompress|resize|noise, level: light|medium|strong}`, `GET /jobs/{id}/attacks`): degraded copy of the *original* → new job linked by `jobs.derived_from` / `jobs.attack` (additive columns), custody entries on both jobs, full re-analysis; comparison = verdict/P/confidence + per-check strongest support (lost / weaker / kept / stronger / new). UI: Attack panel (level + 3 buttons, polling, per-check degradation, open copy); attacked jobs show a banner linking back to the original. "Export PDF report" button.
- **Results table** (`scripts/results_table.py [--attacks] [--long DIR]` → `data/results/results.md|json`): verdict + the system's own explanation per file, measured screening time saving (screening + deadline deep vs unlimited deep), screening recall vs ground truth, pooled control miss-rate, attack degradation.

**Results (data/results/results.md)**
- Verdicts: acceptable **7/7**, exact 6/7 (04 → likely authentic instead of inconclusive).
- Screening recall: **7/7** ground-truth segments inside flagged intervals. Controls: 30 deep-checked, 0 hits → miss-rate 0% raw, 3% smoothed.
- Time saving: test set (8 s, 640×480) **−9%** (adaptive costs more on short clips); long clips (≈55 s, 720p) **+36%** (45.0 s vs 70.5 s; coverage 48% / 86%).
- Attack mode (medium) on the 3 manipulated files: 7/9 copies → inconclusive, **2/9 → likely authentic** (03 recompressed P 0.18, 03 noised P 0.18), 0/9 stayed manipulated. In the live API run (fewer ledger entries) 03's copies stayed inconclusive because lineage to the trusted original was kept; with more versions in the ledger the attacked copy's nearest lineage parent is the untrusted 03 itself, so that evidence turns neutral.

**How to run**
```
curl -o report.pdf "http://localhost:8000/jobs/<id>/report.pdf?actor=alice"
curl -X POST http://localhost:8000/jobs/<id>/attack -H "Content-Type: application/json" -d "{\"kind\":\"recompress\",\"level\":\"medium\"}"
.venv\Scripts\python scripts\results_table.py --attacks [--long <dir>]
.venv\Scripts\python -m pytest -q     # 58 tests
```

**Known issues**
- Attack-induced false negatives (2/9): recompression/noise erase the noise-residual and temporal evidence of the 03 splice and fusion then calls it "likely authentic". A quality gate (refuse "likely authentic" when the file is heavily compressed / noisy, e.g. from `metadata.reencode_generations`, bitrate, or noise level) would turn these into inconclusive — not implemented (fusion change → ask first).
- Lineage parent choice ignores trust when a closer untrusted relative exists, so trust does not flow through derived copies (noted in Step 8).
- PDF pages were checked via text extraction (no PDF renderer on this machine); images are embedded (466 KB report).
- Device signing (stretch #4) and blockchain anchoring (#5) not implemented by choice.

## Step 15 — Polish: known-issue fixes, README, demo fallback, capture-device signing ✅
**Known issues fixed**
- **Interrupted jobs** (Step 2): `runner.recover()` at startup re-runs jobs left in queued/preprocessing/screening/analyzing/deep (custody `resumed_after_restart`); interrupted "analyze deeper" keeps the previous complete results. `process_job` clears stale findings first, so stage-saved findings are never duplicated.
- **Trust lost through derived copies** (Steps 8, 14): lineage findings are judged against the nearest *trusted* relative when one exists (`graph.trusted_relative`); the graph still shows the closest parent. 07 now judged against 01 (P 0.178 → 0.136); attacked copies of 03 keep "0.4 s removed at 6.0 s".
- **Attack false negatives** (Step 14): new neutral whole-file check `quality.media` (x264 crf from the SEI ≥ 30, Immerkaer noise σ ≥ 5, bits/pixel < 0.01 only when crf is unknown, height < 240) + fusion gate: degraded media cannot be "likely authentic" unless a strong contradicting finding exists (e.g. a device-signed or trusted match). Attack results: **0/9 → likely authentic** (was 2/9), 9/9 inconclusive; test-set verdicts unchanged (7/7 acceptable).
- **Vite IPv6-only binding** (Step 7): `backend/serve.py` serves API (`/api`) + built UI (`/`) in one process — no Vite needed for demos.

**Demo fallback**
- `scripts/demo_prepare.py` → self-contained `demo_data/` (gitignored): full pipeline over the test set, 01 signed at capture by simulated `cam01` and trusted, 02 submitted with 01's camera manifest (origin chain broken, localized to audio 2–8 s), recompressed attack copy of 03, PDF reports in `demo_data/reports/`, `MANIFEST.json`, `DEMO.md` (links + talking points). Only deletes a folder carrying its own marker on `--force`.
- `scripts/demo_serve.py [--data demo_data] [--open]` → one-port server, builds the UI if needed, prints deep links, warns if the folder was moved.
- Verified: served `demo_data`, all 8 jobs listed with expected verdicts, UI rendered from the single server (02 evidence: "Origin chain broken: 10/16 segments still match what device 'cam01' signed …").

**Capture-device signing (stretch #4)**
- `backend/provenance/device.py` (Ed25519 device keys, signed manifest = device id, time, SHA-256, Merkle root + leaves; `devices` table), `scripts/capture_device.py keygen|register|sign`, `POST/GET /devices`, `POST /jobs` accepts an optional `manifest`.
- Check `provenance.capture_signature` (`backend/checks/capture.py`): intact (strong clean, capped) / content intact (re-muxed) / broken (localized changed segments → supports) / re-encoded (neutral) / forged signature (supports) / unregistered device (not assessable).

**README**: setup, run modes, 7-minute demo script with exact clicks, fallback procedure, results, device walkthrough, API table, limitations, troubleshooting.

**Tests**: 62 passing (+ quality gate, capture chain states, resume-after-restart without duplicates, single-process server).

**Remaining known issues**
- Thin manipulated margins (P 0.70–0.77 vs 0.70); hand-set weights/thresholds/prior.
- 04 is "likely authentic" (expected inconclusive, acceptable): crf 24 is below the degradation threshold and its lineage to the trusted original is clean.
- Headless-browser video frames stay black (playback to be checked in a normal browser); PDF checked by text extraction only.
- CLAUDE.md says `docs/`; the source PDF/PNG are at the repo root.

## Step 16 — Still images: texture-aware spatial check, still lineage, image-native checks ✅
All changes gated on `media_type == "image"`; no change to video checks, fusion thresholds/weights, scheduler or frontend.

**Safety**: video verdict table before (`data/regression_before.md`) and after (`data/regression_after.md`): all 7 verdicts identical, P equal to ±0.001; the only diff is 05's deep coverage 75% vs 100% (wall-clock deadline variance under load in the baseline run, not a code effect).

**Step 1 — texture-aware spatial check (`backend/checks/visual.py`, still-image path only)**
- Blocks of 16 px (32 px above 2048 px) are skipped when near-black/saturated (mean < 16 or > 239, > 5% clipped) or flat (std < 2 grey levels, where JPEG erases noise); `coverage` = share of assessable blocks.
- Each block's Immerkaer noise (or ELA) is z-scored against blocks of similar brightness (3 bins) × texture (5 bins, texture = std of a σ=2-smoothed image); small strata fall back to texture-only.
- Event = connected region with |z| > 4 and ×1.8 deviation, ≥ max(16 blocks, 1.5% of assessable), mean |z| ≥ 5, fill ≥ 0.35 → supports (reliability 0.45 noise / 0.30 ELA).
- Nothing found → weak `contradicts`, score 0.3, reliability = cap × coverage (cap 0.30 noise, 0.20 ELA); coverage < 20% → neutral "not enough assessable area".
- ELA only for JPEGs, re-saved at the file's own estimated quality; PNG/no JPEG history → not_assessable.
- Old `_region_check` removed (was image-only).

**Step 2 — still-image lineage (`backend/provenance/fingerprint.py`)**: a one-packet video stream is decoded as one frame (no `fps` filter, which emitted nothing) → thumbnails/pHash exist for images, so lineage, Merkle and ledger comparison work. Video command unchanged.

**Step 3 — image-native check (`backend/checks/image.py`, helper `_jpeg.py`; registered for images only)**
- `image.jpeg_quant` (metadata): DQT tables parsed from the file (zigzag → natural), SOF, IJG-quality estimate, exact-standard vs custom table; camera Make/Model + exact libjpeg table → supports (re-saved by software); camera + custom table → weak contradicts; else neutral. Not a JPEG → not assessable.
- `image.double_jpeg` (visual): median periodic-gap score of low-frequency DCT histograms; measured: single compression 0.08–0.12, coarse→fine double 0.78, 3× re-save 0.26 → ≥ 0.40 supports, 0.20–0.40 weak supports, < 0.15 weak contradicts.
- `image.exif_consistency` (metadata): DateTimeOriginal vs ModifyDate, editing-software `Software`, embedded thumbnail vs image (global correlation over 4 rotations + **local difference map**: a compact area ≥ 0.6 std units and ≥ 6× typical → "changed after capture", with region box). No EXIF → not assessable.
- `quality.media` (image branch): estimated JPEG quality < 60 → degraded (quality gate already in fusion).

**Image test set** (`scripts/make_image_testset.py` → `data/image_testset/` + expected.json, base = user's F1 photo downscaled): clean, spliced (pasted region from another photo, EXIF + original thumbnail copied), resaved_3gen, exif_edited, screenshot.png. `scripts/eval_images.py <dirs>` prints the table below.

| file | fired (supports) | assessable | P | verdict | confidence |
|---|---|---|---|---|---|
| clean_still_synthetic.jpg (before: likely manipulated 0.71) | – | 2 | 0.242 | inconclusive | abstained (too few assessable) |
| f1_car.jpg (before: inconclusive 0.250, 0 assessable) | – | 3 | 0.211 | inconclusive | abstained |
| podium_aws.jpg (before: noise false region, 0.417) | – | 2 | 0.221 | inconclusive | abstained (too few assessable) |
| clean.jpg | – | 7 | 0.161 | **likely authentic** | 0.84 |
| exif_edited.jpg | exif_consistency, metadata.encoder | 7 | 0.666 | inconclusive (rose from 0.16) | abstained |
| resaved_3gen.jpg | double_jpeg (+ quality gate) | 3 | 0.252 | inconclusive | abstained |
| screenshot.png | – | 0 | 0.250 | inconclusive | abstained (nothing assessable: PNG, no EXIF) |
| spliced.jpg | exif_consistency (local thumbnail diff inside the paste) | 7 | 0.277 | inconclusive (rose from 0.16) | abstained |
No false "likely manipulated" on any clean image; podium backdrop and F1 photo no longer flagged.

**Known issues / open decision**
- **Splice not caught by pixel statistics**: the pasted region has 2.25× the noise of its surroundings, but natural block-to-block variation in this photo is as large (median z ≈ 1, also with region-level averaging), so noise/ELA stay clean; only the EXIF-thumbnail cue catches it. A splice without a thumbnail would currently read "likely authentic" from consistent copied metadata.
- **Clean web images abstain by design of the prior**: with no EXIF/provenance, metadata and provenance stay at their prior (P ≥ 0.175 from those two alone) and only one modality is assessed (`MIN_MODALITIES = 2`, `MIN_ASSESSABLE = 3`). Proposal (not implemented, needs approval): keep the rule — an image whose origin cannot be checked should be inconclusive — and say so in the reason text ("no capture metadata or provenance to corroborate"); if a decision is wanted for such images, an image-gated `MIN_MODALITIES_IMAGE = 1` would still leave P ≈ 0.21 > T_AUTH, so it would not change outcomes without also lowering T_AUTH for images, which I do not recommend.
- Block-artifact-grid (misaligned 8×8 grid) splice detection was prototyped and dropped: the test base, itself a downscaled JPEG, leaves residual grid aliasing that made clean images fire.
- Tests: 67 passing (+5 image tests).
