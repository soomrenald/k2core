# Extraction baseline

This record freezes the source state used to begin the behavior-preserving
`k2core` extraction on July 22, 2026.

## Source revisions

- Desktop K2Lab: `soomrenald/k2lab` at
  `508f59f9d600af21e775a5c9f10b5631a25928ee`.
- K2Lab RunPod: `soomrenald/k2lab_runpod` at
  `1bd1e1bd656ee8c247e6061347e30590b875c056`.

The desktop checkout contained three pre-existing untracked prompt fixtures:
`prompts/test4.json`, `prompts/testeight.json`, and `prompts/testfive.json`.
They are not part of the extraction and were not modified.

## Test baseline

The desktop suite completed with 169 passed tests, 2 expected GPU-worker skips,
and 6 passed subtests. The RunPod suite was not executed in this session because
another Codex instance was concurrently building its workspace image; this
repository records that source revision without writing to that checkout.

## Duplicate-module inventory

At the revisions above, the following desktop and RunPod source files were
byte-identical:

- `config.py`
- `debug.py`
- `face_detail.py`
- `image_edit.py`
- `memory.py`
- `project.py`
- `projector.py`
- `regional_lora.py`
- `regional_prompting.py`
- `sampling.py`
- `spatial_attention.py`
- the complete `lora/`, `model/`, `regions/`, and `worker/` trees

`output.py` differed only in application-specific workspace discovery text and
fallback path. That module must be split so path resolution remains in consumer
adapters while shared naming and metadata move into `k2core`.

Desktop `app.py`, `desktop/`, `qml/`, and process-launching behavior remain in
K2Lab. RunPod `agent/`, `web/`, React, provider, transport, security, persistence,
and workspace behavior remain outside `k2core` and are candidates for the later
`runpod_core` extraction.

