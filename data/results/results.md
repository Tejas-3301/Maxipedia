# Results on the ground-truth test set

Generated 2026-10-12 13:43 UTC; trusted reference: 01_clean.mp4; verdicts by evidence fusion only.

## Verdicts

| file | ground truth | expected | verdict | P(manip) | confidence | acceptable | system's explanation |
|---|---|---|---|---|---|---|---|
| 01_clean.mp4 | authentic | likely authentic | **likely authentic** | 0.169 | 0.83 | yes (exact) | no evidence for manipulation |
| 02_voice_swap.mp4 | manipulated | likely manipulated | **likely manipulated** | 0.701 | 0.70 | yes (exact) | audio.hf_cutoff (2.4-5.6s); provenance.merkle (2.0-8.0s); sync.lipsync (2.4-5.8s) |
| 03_splice.mp4 | manipulated | likely manipulated | **likely manipulated** | 0.727 | 0.73 | yes (exact) | provenance.lineage (5.9-6.1s); visual.noise_residual (3.0-4.5s); provenance.merkle (1.0-8.0s) |
| 04_reencoded_3x.mp4 | authentic_content_recompressed | inconclusive | **likely authentic** | 0.195 | 0.80 | yes | metadata.reencode_generations |
| 05_metadata_edited.mp4 | manipulated_metadata | likely manipulated | **likely manipulated** | 0.765 | 0.77 | yes (exact) | metadata.timestamps; metadata.encoder |
| 06_derived_a.mp4 | authentic_derivative | likely authentic | **likely authentic** | 0.134 | 0.87 | yes (exact) | no evidence for manipulation |
| 07_derived_b.mp4 | authentic_derivative | likely authentic | **likely authentic** | 0.136 | 0.86 | yes (exact) | no evidence for manipulation |

Acceptable verdict on **7/7** files; exactly the expected verdict on 6/7.

## Screening time saving (measured, test set)

| file | screening s | deep with deadline s | budget s | deep unlimited s | saving | deep coverage | not analysed |
|---|---|---|---|---|---|---|---|
| 01_clean.mp4 | 0.36 | 7.80 | 20 | 7.94 | -3% | 100% | 0 (0 flagged) |
| 02_voice_swap.mp4 | 0.35 | 5.13 | 20 | 5.03 | -9% | 100% | 0 (0 flagged) |
| 03_splice.mp4 | 0.34 | 5.09 | 20 | 5.01 | -8% | 100% | 0 (0 flagged) |
| 04_reencoded_3x.mp4 | 0.34 | 6.56 | 20 | 6.54 | -5% | 100% | 0 (0 flagged) |
| 05_metadata_edited.mp4 | 0.35 | 6.63 | 20 | 7.69 | +9% | 100% | 0 (0 flagged) |
| 06_derived_a.mp4 | 0.26 | 4.24 | 20 | 4.29 | -5% | 100% | 0 (0 flagged) |
| 07_derived_b.mp4 | 0.24 | 4.15 | 20 | 4.15 | -6% | 100% | 0 (0 flagged) |

Total: screening + deadline-limited deep analysis **41.8 s** vs unlimited deep analysis **40.6 s** → measured saving **-3%** (negative = adaptive costs more).

## Screening recall and miss-rate

- Ground-truth manipulated segments inside a *flagged* interval: **7/7** (recall 100%).
- Control intervals deep-checked (unflagged time): **21**, with deep findings: **0** → raw miss-rate 0%, Laplace-smoothed 4% (the value fusion uses to cap clean evidence).

## Attack mode (medium) — evidence degrading

| file | attack | verdict before | P before | verdict after | P after | remaining top evidence |
|---|---|---|---|---|---|---|
| 02_voice_swap.mp4 | re-encode H.264 crf 35, AAC 64k | likely manipulated | 0.701 | **inconclusive** | 0.637 | audio.noise_floor, sync.lipsync, audio.hf_cutoff |
| 02_voice_swap.mp4 | downscale to 50% then re-encode | likely manipulated | 0.701 | **inconclusive** | 0.673 | audio.hf_cutoff, sync.lipsync, audio.noise_floor |
| 02_voice_swap.mp4 | temporal video noise strength 12, white audio noise amplitude 0.01 | likely manipulated | 0.701 | **inconclusive** | 0.261 | sync.lipsync |
| 03_splice.mp4 | re-encode H.264 crf 35, AAC 64k | likely manipulated | 0.727 | **inconclusive** | 0.413 | provenance.lineage |
| 03_splice.mp4 | downscale to 50% then re-encode | likely manipulated | 0.727 | **inconclusive** | 0.457 | provenance.lineage, visual.temporal |
| 03_splice.mp4 | temporal video noise strength 12, white audio noise amplitude 0.01 | likely manipulated | 0.727 | **inconclusive** | 0.412 | provenance.lineage |
| 05_metadata_edited.mp4 | re-encode H.264 crf 35, AAC 64k | likely manipulated | 0.765 | **inconclusive** | 0.253 | metadata.encoder |
| 05_metadata_edited.mp4 | downscale to 50% then re-encode | likely manipulated | 0.765 | **inconclusive** | 0.253 | metadata.encoder |
| 05_metadata_edited.mp4 | temporal video noise strength 12, white audio noise amplitude 0.01 | likely manipulated | 0.765 | **inconclusive** | 0.253 | metadata.encoder |

9 attacked copies of manipulated files: 0 flipped to 'likely authentic', 9 became inconclusive, 0 stayed 'likely manipulated'.
