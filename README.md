# Visuo-Haptic Texture Rendering (Real-Time, Windows-First)

## Overview

- Goal: Render realistic micro-texture vibrations (and optional audio) when a pointer/stylus moves over a textured 3D object.
- Stack: Python (PyTorch) for training → ONNX Runtime (C++) for real-time inference → CHAI3D for 3D + UV picking → PortAudio/Arduino for actuator output.

## Features

- Data-driven vibrotactile synthesis conditioned on texture patch + motion (speed/force)
- Optional co-generated audio for synchronized sound + feel
- Real-time C++ runtime with ONNX Runtime + CHAI3D (Windows)
- Low-latency output via PortAudio (audio DAC path) or DRV2605 driver (Arduino path)
- Strong debugging: TensorBoard logs, spectrogram plots, WAV dumps, failure artifacts

Tooling links: [CHAI3D](https://www.chai3d.org/), [ONNX Runtime (C++)](https://onnxruntime.ai/docs/get-started/with-cpp.html), [PortAudio](https://www.portaudio.com/).

## Datasets

Primary (recommended): Tactile, Audio, and Visual Dataset During Bare Finger Interaction with Textured Surfaces (2025)

- Article: https://www.nature.com/articles/s41597-025-04670-0
- Data (Figshare): https://springernature.figshare.com/articles/dataset/26965267

Contents: stereo texture images, fingertip position/speed/force, emitted sound, friction-induced vibrations. Perfect for training image+action → vibro(+audio).

(You can still plug in other visuo-tactile sets later; the loaders are generic.)

## Model Options (pick one to start)

- Baseline (fast): CNN encoder (texture) + state MLP (speed/force) → TCN/Transformer decoder → 100 ms vibro @ 1 kHz
- Low-delay (snappier): add prev_vib history to inputs and a light attention block (reacts faster to motion)
- Multitask (audio+vibro): shared encoder → dual decoders (vibro 1 kHz; audio 8–16 kHz) for synchronized rendering

Research to cite:
- Image→vibration cGAN (Ujitoko & Ban): https://arxiv.org/abs/1902.07480
- Action-conditional low-latency models validated for real-time haptics (Heravi et al., ToH 2024): https://dl.acm.org/doi/abs/10.1109/TOH.2024.3382258 and https://arxiv.org/abs/2212.13332

## Repository Layout

```
/python
  environment-gpu.yml         # conda env (CUDA)
  environment-cpu.yml         # conda env (CPU-only)
  dataset_textures.py         # generic visuo-tactile dataset loader + sanity CLI
  losses.py                   # time/STFT/mel/envelope-sync losses
  metrics.py                  # RMSE, LSD, centroid, envelope corr
  model_texturenet.py         # baseline + low-delay + multitask variants
  train.py                    # rich logging; sanity/overfit modes; checkpoints
  export_onnx.py              # export + validate ONNX vs PyTorch
  utils.py                    # seeds, plots, WAV helpers, exception hook
  params.yaml                 # config (fs, bands, model/loss/training)
/cpp
  CMakeLists.txt
  src/
    main.cpp                  # CHAI3D window, input loop, glue
    UvPicking.*               # ray → triangle → barycentric → UV
    TextureSampler.*          # UV → 96×96 patch (normalized)
    OnnxRunner.*              # load & run model.onnx
    RingBuffer.h              # SPSC ring for audio/vibro streams
    AudioOut.*                # PortAudio output (WASAPI/ASIO)
/assets
  dataset.parquet (your preprocessed data; not committed)
  model.onnx
  norm.yaml
  textures/...
/docs
  RUN_WINDOWS.md              # step-by-step Windows build notes
/scripts
  build.ps1                   # CMake configure + build
```

## Quickstart (Conda)

Create env (GPU):

```
conda env create -f python/environment-gpu.yml
conda activate visuotactile-gpu
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Or CPU-only:

```
conda env create -f python/environment-cpu.yml
conda activate visuotactile-cpu
```

## Data Sanity & Preprocessing

Place your preprocessed table at `assets/dataset.parquet`.

Sanity check a few windows and plots:

```
python -m python.dataset_textures --data assets/dataset.parquet --split train --inspect 32 --out assets/debug_samples
```

This saves a grid of texture patches, speed/force histograms, and vibro/audio PSD plots to help catch data issues early.

## Train

Baseline vibro (1 kHz):

```
python -m python.train --data assets/dataset.parquet --mode baseline_vibro --epochs 80
```

Low-delay vibro (adds prev_vib):

```
python -m python.train --data assets/dataset.parquet --mode low_delay_vibro --epochs 100
```

Multitask (vibro+audio):

```
python -m python.train --data assets/dataset.parquet --mode multitask_av --epochs 100
```

Logs: `runs/` (TensorBoard), plus rich console; failure artifacts in `assets/debug/`.

Losses combine time MSE + STFT for vibro; audio head adds mel/STFT and an envelope-synchrony term (keeps sound+feel in sync).

Use `--sanity_check` and set `overfit_batches` in `python/params.yaml` when debugging.

## Export to ONNX

```
python -m python.export_onnx --ckpt assets/ckpt/best.pt --mode baseline_vibro --out assets/model.onnx
```

The script also runs a small batch through onnxruntime to verify numerical parity with PyTorch before you ship the model. Reference: [ONNX Runtime docs](https://onnxruntime.ai/docs/get-started/with-cpp.html).

## Runtime App (Windows)

- 3D + picking: CHAI3D loads your UV-mapped mesh and computes the contact UV each frame. See: https://www.chai3d.org/
- Inference: ONNX Runtime (C++) runs the exported model; stream 10–20 ms chunks into a ring buffer. See: https://onnxruntime.ai/
- Output paths:
  - Recommended: Audio DAC path via PortAudio → small class-D amp → LRA/ERM on the stylus (smooth timing, great fidelity). Downloads: https://files.portaudio.com/download.html
  - Alternative: Arduino + DRV2605 LRA/ERM haptic driver (USB serial packets; auto-resonance tracking). Product page: https://www.ti.com/product/DRV2605 · Datasheet: https://www.ti.com/lit/ds/symlink/drv2605.pdf

Build notes are in `docs/RUN_WINDOWS.md` (set `CHAI3D_DIR`, `ONNXRUNTIME_DIR`, `PORTAUDIO_DIR` in CMake). PortAudio main site: https://www.portaudio.com/

## Evaluation Ideas

- Objective: RMSE (time) & log-spectral distance for vibro; mel/STFT metrics for audio; envelope correlation for synchrony
- Subjective: 7-pt realism; ABX discrimination (wood vs stone vs fabric)
- Latency: cursor-speed threshold → output pulse timing; target < 50 ms end-to-end

## Hardware Notes (Quick BoM)

- Actuator: LRA (~170–200 Hz resonance) or ERM coin motor
- Driver (MCU path): TI DRV2605/DRV2605L haptic drivers (auto-resonance, waveform library)
- Audio path: USB audio interface + small class-D amp if you prefer DAC streaming

## Troubleshooting

- NaNs/exploding loss: lower LR, disable AMP (`amp: false`), reduce `tcn_blocks`, run with `overfit_batches=1` in params, check debug WAVs/plots in `assets/debug/`
- GPU OOM: cut batch size (`train.batch` in `python/params.yaml`), crop size, or decoder width; try CPU env to isolate driver/CUDA issues
- ONNX mismatch: re-export with the same `params.yaml`; ensure opset ≥ 17; see ORT C++ docs: https://onnxruntime.ai/
- No audio output: verify PortAudio back-end (WASAPI Exclusive/ASIO), reduce frames-per-buffer (64–128), check device sample rate: https://www.portaudio.com/

## Roadmap

- Add perceptual roughness auxiliary head (expose a “roughness” slider)
- Precompute “haptic maps” for ultra-low runtime cost (AM/FM from per-pixel stats)
- Extend to stylus-specific data and anisotropic textures

## Attributions & Links

- Dataset (Sci Data 2025): https://www.nature.com/articles/s41597-025-04670-0 · Data: https://springernature.figshare.com/articles/dataset/26965267
- CHAI3D: https://www.chai3d.org/ (About/devices: https://www.chai3d.org/concept/about)
- ONNX Runtime (C++): https://onnxruntime.ai/docs/api/c/ · https://onnxruntime.ai/docs/get-started/with-cpp.html
- PortAudio: https://www.portaudio.com/ · Downloads: https://files.portaudio.com/download.html
- TI DRV2605: https://www.ti.com/product/DRV2605 · Datasheet: https://www.ti.com/lit/ds/symlink/drv2605.pdf
- Key papers: cGAN vibro https://arxiv.org/abs/1902.07480 · action-conditional ToH’24 https://dl.acm.org/doi/abs/10.1109/TOH.2024.3382258 / https://arxiv.org/abs/2212.13332

