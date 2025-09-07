from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort

from .model_texturenet import TextureNet
from .train import load_params


def export_model(ckpt_path: Path, mode: str, params_path: Path, out_path: Path) -> int:
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    params = load_params(params_path)
    mcfg = params.get("model", {})
    use_prev = bool(mcfg.get("use_prev", False) or mode == "low_delay_vibro")
    net = TextureNet(
        mode=mode,
        latent=int(mcfg.get("latent", 128)),
        tcn_blocks=int(mcfg.get("tcn_blocks", 4)),
        tcn_growth=int(mcfg.get("tcn_growth", 64)),
        use_prev=use_prev,
    ).eval()

    state = torch.load(ckpt_path, map_location="cpu")
    net.load_state_dict(state["model"])

    # Dummy inputs
    B = 4
    patch = torch.randn(B, 1, 96, 96)
    state_in = torch.rand(B, 2)
    inputs = [patch, state_in]
    input_names = ["patch", "state"]
    dynamic_axes = {"patch": {0: "B"}, "state": {0: "B"}}
    if use_prev:
        prev_vib = torch.randn(B, 100)
        inputs.append(prev_vib)
        input_names.append("prev_vib")
        dynamic_axes["prev_vib"] = {0: "B"}

    if mode == "multitask_av":
        output_names = ["vib", "audio"]
    else:
        output_names = ["vib"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        net,
        tuple(inputs),
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=17,
        dynamic_axes=dynamic_axes,
    )

    # Validate with onnxruntime
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])  # portable
    feeds = {"patch": patch.numpy(), "state": state_in.numpy()}
    if use_prev:
        feeds["prev_vib"] = inputs[-1].numpy()
    ort_outs = sess.run(None, feeds)

    with torch.no_grad():
        pt_out = net(*inputs)
        if mode == "multitask_av":
            pt_out = [pt_out[0].numpy(), pt_out[1].numpy()]
        else:
            pt_out = [pt_out.numpy()]
    max_diffs = [float(np.max(np.abs(a - b))) for a, b in zip(ort_outs, pt_out)]
    print({"onnx_outputs": [o.shape for o in ort_outs], "max_abs_diff": max_diffs})
    assert all(d <= 1e-4 for d in max_diffs), f"ONNX validation failed: diffs={max_diffs}"
    return 0


def main():
    import argparse
    p = argparse.ArgumentParser(description="Export TextureNet to ONNX")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["baseline_vibro", "low_delay_vibro", "multitask_av"])
    p.add_argument("--params", type=str, default="params.yaml")
    p.add_argument("--out", type=str, default="assets/model.onnx")
    args = p.parse_args()
    sys.exit(export_model(Path(args.ckpt), args.mode, Path(args.params), Path(args.out)))


if __name__ == "__main__":
    main()

