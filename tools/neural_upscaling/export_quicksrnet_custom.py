#!/usr/bin/env python3
"""Export custom-shape QuickSRNetSmall research assets.

This wrapper exists because Qualcomm's official Android-ready export path needs
AI Hub Workbench access. When Workbench is unavailable, use --local-onnx-only to
prove a proposed input shape traces correctly before submitting a cloud export.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export custom-shape QuickSRNetSmall assets for GameNative research.",
    )
    parser.add_argument("--height", type=int, default=180, help="Input height before 4x upscale.")
    parser.add_argument("--width", type=int, default=320, help="Input width before 4x upscale.")
    parser.add_argument("--scale-factor", type=int, default=4, help="QuickSRNet scale factor.")
    parser.add_argument(
        "--runtime",
        choices=[
            "tflite",
            "qnn_dlc",
            "qnn_context_binary",
            "onnx",
            "precompiled_qnn_onnx",
        ],
        default="tflite",
        help="Qualcomm AI Hub target runtime for official export.",
    )
    parser.add_argument(
        "--precision",
        choices=["float", "w8a8"],
        default="float",
        help="Export precision. Keep float first while validating image quality.",
    )
    parser.add_argument(
        "--device",
        default="Samsung Galaxy S25 (Family)",
        help="AI Hub device name used for official export.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_assets") / "quicksrnet_custom",
        help="Directory for exported artifacts.",
    )
    parser.add_argument(
        "--local-onnx-only",
        action="store_true",
        help="Only trace a local ONNX file. Does not require Qualcomm AI Hub access.",
    )
    parser.add_argument(
        "--skip-profiling",
        action="store_true",
        help="Skip hosted-device profiling during official export.",
    )
    parser.add_argument(
        "--skip-inferencing",
        action="store_true",
        help="Skip hosted-device inference comparison during official export.",
    )
    return parser.parse_args()


def validate_shape(height: int, width: int, scale_factor: int) -> None:
    if scale_factor != 4:
        raise SystemExit("This research path currently expects QuickSRNetSmall scale factor 4.")
    if height <= 0 or width <= 0:
        raise SystemExit("Height and width must be positive.")
    if height % 4 != 0 or width % 4 != 0:
        raise SystemExit("Use input dimensions divisible by 4 to keep texture paths simple.")


def local_onnx_export(height: int, width: int, scale_factor: int, output_dir: Path) -> Path:
    from qai_hub_models.models.quicksrnetsmall.model import QuickSRNetSmall

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"quicksrnetsmall_x{scale_factor}_{width}x{height}_to_{width * scale_factor}x{height * scale_factor}.onnx"

    print(f"Loading QuickSRNetSmall x{scale_factor} weights...")
    model = QuickSRNetSmall.from_pretrained(scale_factor=scale_factor).eval().to("cpu")
    sample = torch.rand(1, 3, height, width, dtype=torch.float32)

    print(f"Tracing local ONNX: input=[1,3,{height},{width}] output=[1,3,{height * scale_factor},{width * scale_factor}]")
    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["image"],
        output_names=["upscaled_image"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print(f"Wrote {output_path}")
    return output_path


def has_ai_hub_access() -> bool:
    from qai_hub_models.utils.qai_hub_helpers import can_access_qualcomm_ai_hub

    return bool(can_access_qualcomm_ai_hub())


def official_export(args: argparse.Namespace) -> None:
    if not has_ai_hub_access():
        raise SystemExit(
            "Qualcomm AI Hub Workbench is not configured, so an official custom "
            "TFLite/QNN export cannot be submitted yet.\n"
            "Run: qai-hub configure --api_token YOUR_TOKEN\n"
            "For now, use --local-onnx-only to validate the custom shape locally."
        )

    command = [
        sys.executable,
        "-m",
        "qai_hub_models.models.quicksrnetsmall.export",
        "--target-runtime",
        args.runtime,
        "--precision",
        args.precision,
        "--scale-factor",
        str(args.scale_factor),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--device",
        args.device,
        "--output-dir",
        str(args.output_dir),
    ]
    if args.skip_profiling:
        command.append("--skip-profiling")
    if args.skip_inferencing:
        command.append("--skip-inferencing")

    print("Running official Qualcomm export:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    validate_shape(args.height, args.width, args.scale_factor)

    if args.local_onnx_only:
        local_onnx_export(args.height, args.width, args.scale_factor, args.output_dir)
        return

    official_export(args)


if __name__ == "__main__":
    main()
