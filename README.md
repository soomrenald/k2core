# K2 Core

K2 Core is the shared, application-neutral Python implementation used by K2Lab,
K2Lab RunPod, and the Wan2Lab product family. It owns reusable image-generation,
regional prompting, adapter routing, worker contracts, memory policy, asset, and
runtime behavior. Product presentation and RunPod-provider behavior remain in
their respective repositories.

The initial `0.1.x` work is a behavior-preserving extraction from K2Lab. Public
APIs can expand during that extraction, but changes must retain compatibility
fixtures and must not introduce Qt, FastAPI, React, RunPod, or Wan-specific
dependencies.

## Development

K2 Core targets Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest
```

The first extracted slice is `k2core.regions`, copied from the matching K2Lab
desktop and RunPod implementations after verifying that those implementations
were identical.

