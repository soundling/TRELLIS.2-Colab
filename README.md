# TRELLIS.2 Colab API

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/soundling/TRELLIS.2-Colab/blob/main/trellis2_l4_ngrok_api.ipynb)
[![Validate notebook and API](https://github.com/soundling/TRELLIS.2-Colab/actions/workflows/validate.yml/badge.svg)](https://github.com/soundling/TRELLIS.2-Colab/actions/workflows/validate.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Turn a reference image into a textured 3D model using **TRELLIS.2 on a Google Colab GPU**, then download it through an authenticated HTTPS API powered by ngrok.

The notebook is self-contained: it installs an isolated Python runtime, loads the models, starts a FastAPI server, opens the tunnel, and includes upload/download helpers. **Only the `.ipynb` is needed in Colab.** The other files make the implementation easy to review and test.

This is a community notebook and API wrapper for [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2), originally used to create assets for Oni Engine. The resulting GLBs can also be opened in Blender or other compatible tools.

## What it does

- Image-to-3D generation with embedded PBR textures in a single `.glb` file.
- L4-oriented low-VRAM configuration: **512 resolution, 30,000 target triangles, 2K textures** by default.
- Optional `1024_cascade` generation, enabled explicitly after testing 512.
- Bearer-token authentication on every route, including health and OpenAPI.
- One GPU job at a time, with at most four outstanding jobs.
- Job status, stage reporting, checksummed downloads, and generation manifests.
- Optional Google Drive persistence for completed jobs.
- Live installation output, 20-second progress heartbeats, and saved installation logs.

This notebook accepts reference images. It does not expose text-to-3D, rigging/animation, or retexturing an existing mesh. Triangle counts are targets, and generated meshes still need scale, topology and material inspection before use in a game.

## Quick start

1. Click **Open in Colab** above and save your own copy if desired.
2. Choose **Runtime → Change runtime type → L4 GPU**. High system RAM is recommended. This installer rejects T4 and requires at least 40 GiB free disk space. It selects or installs the CUDA 12.6 toolkit to match the pinned PyTorch wheels, including when Colab defaults to CUDA 13. It creates a Python 3.11 environment without replacing the notebook kernel's packages.
3. Create an [ngrok account and authtoken](https://dashboard.ngrok.com/get-started/your-authtoken).
4. Accept the access conditions for [DINOv3 ViT-L on Hugging Face](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), then create a read token with access to that model.
5. Add these values in Colab's **Secrets** panel and enable notebook access:

   | Secret | Purpose |
   |---|---|
   | `NGROK_AUTHTOKEN` | Opens your ngrok tunnel |
   | `HF_TOKEN` | Downloads the required model weights |
   | `TRELLIS_API_KEY` | Authenticates your API clients; at least 32 random ASCII characters |

   Generate an API key privately, for example with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Never put its value in a committed cell or shared output.

6. Run notebook sections **1–7 in order**. Initial extension compilation and weight downloads can take a while; zero GPU usage during CPU compilation is normal. Wait for `Model ready` and the successful external health check.
7. Copy the printed **API URL**. Keep the notebook session active while using it. Enable the optional smoke-test or reference-upload cell to generate a first model.
8. When finished, download your assets, set `STOP_SERVICE = True` in the final cell and run it, then disconnect the runtime.

`USE_GOOGLE_DRIVE = True` stores jobs under `MyDrive/Trellis2Oni/jobs`. Otherwise, `/content` results are temporary. The tunnel URL may change after reconnecting.

## Call it from your computer

Clone this repository and install the small client dependency:

```bash
git clone https://github.com/soundling/TRELLIS.2-Colab.git
cd TRELLIS.2-Colab
python -m pip install requests
```

Set `TRELLIS_URL` to the HTTPS URL printed by Colab and `TRELLIS_API_KEY` to the same private key stored in Colab. Supply these through your local environment; do not commit their values.

```python
import os
from client import submit_asset, download_job

url = os.environ['TRELLIS_URL']
key = os.environ['TRELLIS_API_KEY']

job_id = submit_asset(
    url, key, 'reference.png',
    seed=42, pipeline='512', steps=12,
    face_count=30000, texture_size=2048,
    remove_background=True, remesh=True,
)
download_job(url, key, job_id, 'model.glb')
```

The client polls job stages, verifies the GLB's SHA-256, and saves a matching `model.manifest.json`. If polling times out, call `download_job` again with the same job ID instead of submitting a duplicate.

For best results, use a clear image of one complete object, with minimal occlusion and a simple background. A transparent PNG with a clean foreground skips automatic background removal.

### curl examples

These examples use Bash environment syntax; on Windows, use WSL/Git Bash or the Python client above.

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $TRELLIS_API_KEY" \
  -H 'ngrok-skip-browser-warning: 1' \
  "$TRELLIS_URL/health"

curl --fail-with-body \
  -H "Authorization: Bearer $TRELLIS_API_KEY" \
  -H 'ngrok-skip-browser-warning: 1' \
  -F 'image=@reference.png' -F 'pipeline=512' -F 'texture_size=2048' \
  "$TRELLIS_URL/jobs"
```

## API

Every request requires `Authorization: Bearer <your key>`. The notebook validates both an authenticated external request and rejection of an anonymous request before reporting the tunnel ready.

| Method | Route | Result |
|---|---|---|
| `GET` | `/health` | Readiness, queue counts, startup error and available profiles |
| `POST` | `/jobs` | Multipart `image` plus settings; returns HTTP 202 and job ID |
| `GET` | `/jobs` | List retained jobs |
| `GET` | `/jobs/{id}` | Status, current stage, settings and result metadata |
| `GET` | `/jobs/{id}/model.glb` | Download a completed model |
| `GET` | `/jobs/{id}/manifest.json` | Download settings, hashes and model provenance |
| `DELETE` | `/jobs/{id}` | Delete a completed or failed job and its files |
| `GET` | `/openapi.json` | Machine-readable API schema |

| Setting | Default | Accepted values |
|---|---|---|
| `pipeline` | `512` | `512`, or `1024_cascade` when `ALLOW_1024 = True` |
| `seed` | `42` | Integer from 0 to 2147483647 |
| `steps` | `12` | Integer from 4 to 30 |
| `face_count` | `30000` | Target from 1000 to 200000 |
| `texture_size` | `2048` | 1024, 2048 or 4096 |
| `remove_background` | `true` | Boolean |
| `remesh` | `true` | Boolean |

The request body is limited to 16 MiB. Images must be at least 32×32 pixels and at most 20 megapixels. Active jobs cannot be deleted or cancelled; wait for completion. A full queue returns **429**, invalid input **422**, and insufficient output disk space **507**.

## Troubleshooting

- **“Unable to locate package cuda-toolkit-12-4”:** that package is absent from NVIDIA's Ubuntu 24.04 repository. Reopen the latest notebook from GitHub and run sections 1–2, then continue in order. It now uses CUDA 12.6 and explicitly pinned PyTorch `2.6.0+cu126` / torchvision `0.21.0+cu126` wheels in `venv-cu126`. NVIDIA's [Ubuntu 24.04 package index](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/) includes `cuda-toolkit-12-6`, and [PyTorch lists this wheel combination](https://docs.pytorch.org/get-started/previous-versions/). Saved Colab copies do not update automatically.
- **“This pinned build needs a CUDA 12.x compiler”:** upload the updated `trellis2_l4_ngrok_api.ipynb` and run sections 1–2 again, then continue in order. Older copies rejected Colab's default compiler. The updated installer reuses CUDA 12.6 if present or installs `cuda-toolkit-12-6`, then explicitly selects its compiler and libraries for the service. NVIDIA documents this as a [toolkit-only package without a driver](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html#meta-packages).
- **CUDA toolkit package installation fails:** inspect `/content/trellis2-service/install.log` for the APT error. The automatic installation uses the CUDA packages available from Colab's configured APT repositories.

| Symptom | What to check |
|---|---|
| Installation appears idle | Read the named command and elapsed-time heartbeat. Builds can use CPU while GPU memory stays empty. Logs: `/content/trellis2-service/install.log` and `install-status.json`. |
| `KeyboardInterrupt` | The cell was interrupted; inspect the last command before rerunning. Keep the session to reuse downloads. |
| Hugging Face 401/403 | Accept the DINOv3 conditions using the same account as `HF_TOKEN`, and check the token's read access. |
| Server disappears while loading | Check system RAM and `server.log`; model offloading still uses CPU memory. |
| CUDA out of memory | Start with 512, 1K textures and `remesh=False`. Restart the runtime if CUDA remains unhealthy. |
| CUDA compiler/version or undefined-symbol error | Start a compatible fresh runtime. Update the PyTorch/CUDA/extension pins together rather than mixing wheels. |
| ngrok authentication/session error | Check the authtoken and account tunnel limits. |
| API 401 | Use `TRELLIS_API_KEY`, not the ngrok authtoken. |
| Client timeout | Retain the job ID and resume polling/download. |
| Session ended | Re-run setup and use the new URL. Only jobs saved to Drive survive deletion of `/content`. |

## Limits and verification

The notebook targets interactive asset generation during an active Colab session, not unattended production hosting. Runtime availability, GPU allocation and session duration vary. Follow [Colab's usage restrictions](https://research.google.com/colaboratory/faq.html); this project includes no keep-alive or timeout-bypass mechanism.

This release preserves all cell sources from the maintainer's September 29, 2026 notebook snapshot. Saved outputs and session metadata were removed. The snapshot was supplied from an existing run; packaging validation does **not** certify a fresh Colab/CUDA installation or new live GPU inference. Use the notebook's preflight and optional real-generation smoke test in your own session. The notebook's original validation note describes its initial development checks.

CPU-only tests cover notebook schema/syntax, clean outputs, embedded-source synchronization, authentication, input limits, queue bounds, single-worker execution, downloads, failures, interrupted-job recovery, GLB validation and installer output/redaction. They use a simulated generation backend; CI downloads no model weights and opens no tunnel.

## Development

The **notebook is the source of truth**. `backend.py`, `api_server.py`, `install_runner.py` and `client.py` are extracted copies for review and testing. Edit the relevant notebook cell, then synchronize them:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python tools/sync_notebook.py
python tools/sync_notebook.py --check
python -m unittest discover -s tests -v
```

Before committing a Colab download, clear its outputs and execution counts and remove session metadata. The tests reject saved outputs, known token patterns and live ngrok URLs. Never attach tokens, private input images, model weights or runtime logs to an issue.

## Credits and licensing

The notebook/API wrapper is [MIT licensed](LICENSE). It downloads, rather than vendors, the underlying models and CUDA dependencies. Their licenses remain separate:

- [Microsoft TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and [TRELLIS.2-4B weights](https://huggingface.co/microsoft/TRELLIS.2-4B).
- [Meta DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), with its own access conditions and license.
- [NVIDIA nvdiffrast](https://github.com/NVlabs/nvdiffrast) and [nvdiffrec](https://github.com/NVlabs/nvdiffrec), with their respective licenses.
- [CuMesh](https://github.com/JeffreyXiang/CuMesh), [FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM), [rembg](https://github.com/danielgatis/rembg) and [pyngrok](https://github.com/alexdlaird/pyngrok).

The wrapper license does not replace model/dependency terms or grant rights to input images.
