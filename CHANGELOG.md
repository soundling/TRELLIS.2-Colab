# Changelog

## 2026-10-02

- Correct the toolkit and PyTorch pins to CUDA 12.6 / cu126 for Colab's Ubuntu 24.04 runtime, whose NVIDIA repository does not contain `cuda-toolkit-12-4`.
- Use a separate `venv-cu126` environment, pin the CUDA wheel variants explicitly, and verify PyTorch's CUDA version before building extensions.
- Select or install CUDA 12.4 to match the pinned PyTorch cu124 wheels instead of rejecting Colab runtimes with a different default compiler.
- Set the service's CUDA compiler, toolkit, executable and library paths explicitly; reuse an existing matching toolkit on reruns.
- Add CPU-only regression coverage for newer/missing compilers, toolkit reuse, and failed installation.

## 2026-09-29

- Initial shareable release from the maintainer's latest `trellis2_l4_ngrok_api.ipynb`.
- Preserved all 25 cell sources; removed eight saved outputs, execution counts and session metadata.
- Added Colab launch link, API/setup documentation, local client, extracted implementation, tests and CPU-only CI.
- Kept TRELLIS.2 code pinned to `75fbf0183001ed9876c8dbb35de6b68552ee08bd` and the notebook's existing package pins.

Original input notebook SHA-256:
`e5bfc915115da9ee3c14706b57fcff51708cd8d04e1a9c7e496c175e7a232b3b`.

The original executed notebook is not distributed. No model weights, generated assets, service credentials or live tunnel URLs are included.
