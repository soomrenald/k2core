# Third-party notices

Last reviewed: 2026-07-30

This file records the third-party software boundaries of `k2core`. It is an engineering
inventory, not a legal opinion or a replacement for upstream license texts.
`pyproject.toml` is the authoritative direct-dependency record.

## Direct runtime dependencies

| Component | Upstream | License recorded by upstream/package metadata |
| --- | --- | --- |
| NumPy | <https://github.com/numpy/numpy> | BSD-3-Clause plus separately identified bundled components; see the installed distribution |
| Pillow | <https://github.com/python-pillow/Pillow> | MIT-CMU |

Native K2 consumers supply PyTorch (BSD-3-Clause), Diffusers (Apache-2.0),
Transformers (Apache-2.0), and safetensors (Apache-2.0). Their exact versions and
license payloads are owned by each consuming application's lockfile or container
manifest.

## Source and model boundaries

No ComfyUI source is copied or vendored in this repository. The legacy adapter calls a
runtime object supplied by a consumer; native inference uses upstream library APIs and
independently maintained K2 code.

This repository does not include or grant rights to Krea, Qwen, LoRA, detector, or
upscaler weights. A consumer that downloads or bundles such assets must review and
preserve each model's separate license, attribution, and redistribution terms.

## First-party license

The `k2core` source is licensed under Apache-2.0; see `LICENSE`. That license covers
first-party software only. It does not grant rights to the separately supplied model
weights or other user assets described above.

Before distribution, generate a complete bill of materials from the final environment
and include every dependency's license and required notice files.
