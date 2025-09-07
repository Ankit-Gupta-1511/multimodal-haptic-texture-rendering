Setup and Training Guide (Conda)

Environments

- GPU (CUDA 12.1):
  - Create: conda env create -f python/environment-gpu.yml
  - Activate: conda activate visuotactile-gpu

- CPU-only:
  - Create: conda env create -f python/environment-cpu.yml
  - Activate: conda activate visuotactile-cpu

Verify Install

- Torch & CUDA: python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Dataset Sanity Inspect

- Run: python -m python.dataset_textures --data assets/dataset.parquet --split train --inspect 32 --out assets/debug_samples
- Outputs: patch grid, speed/force histograms, vib/audio PSDs, printed stats.

Quick Sanity Train

- Baseline vibro: python -m python.train --data assets/dataset.parquet --mode baseline_vibro --epochs 2 --sanity_check 1
- Low-delay vibro: python -m python.train --data assets/dataset.parquet --mode low_delay_vibro --epochs 2 --sanity_check 1
- Multitask A/V: python -m python.train --data assets/dataset.parquet --mode multitask_av --epochs 2 --sanity_check 1

Export ONNX

- Baseline: python -m python.export_onnx --ckpt assets/ckpt/best.pt --mode baseline_vibro --out assets/model.onnx
- Low-delay: python -m python.export_onnx --ckpt assets/ckpt/best.pt --mode low_delay_vibro --out assets/model.onnx
- Multitask: python -m python.export_onnx --ckpt assets/ckpt/best.pt --mode multitask_av --out assets/model.onnx

TensorBoard

- Start: tensorboard --logdir runs

Troubleshooting

- CUDA OOM: reduce batch size (train.batch), disable AMP (train.amp=false), reduce tcn_blocks or tcn_growth in params.yaml.
- NaNs/Inf: check dataset integrity; dataset CLI helps find issues. Training logs will show if loss explodes—lower LR and ensure bandpass filter is correct.
- Kernel/driver mismatch: make sure NVIDIA driver ≥ 531.xx for CUDA 12.1; recreate env if PyTorch/CUDA wheels mismatch.
