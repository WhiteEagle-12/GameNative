# Neural Upscaling Research Tools

This folder contains helper tooling for exporting custom-shape QuickSRNetSmall
assets for the GameNative neural upscaling research branch.

## Why this exists

The default QuickSRNetSmall TFLite model we tested is fixed at:

```text
input:  128x128 RGB
output: 512x512 RGB
scale:  4x
```

Qualcomm's AI Hub Models exporter exposes custom input shapes through:

```text
--height
--width
--scale-factor
```

The model input spec is NCHW float RGB:

```text
image: [batch, 3, height, width]
```

For Android TFLite assets, Qualcomm's official exporter compiles through
Qualcomm AI Hub Workbench. That requires a configured API token:

```powershell
qai-hub configure --api_token YOUR_TOKEN
```

Without Workbench access, the official exporter can fetch static release assets,
but those static assets do not represent our custom dimensions.

## Recommended First Shape

Start with:

```text
input:  320x180
output: 1280x720
scale:  4x
```

This is large enough to be meaningful for games while still small enough that
the Odin's early 128x128 timing suggests it may be practical.

## Local ONNX Shape Sanity Check

This does not produce an Android-ready model, but it proves the custom shape can
be traced locally:

```powershell
uv venv --python 3.12 .venv-qai
uv pip install --python .\.venv-qai\Scripts\python.exe qai-hub-models
uv pip install --python .\.venv-qai\Scripts\python.exe onnxscript
.\.venv-qai\Scripts\python.exe tools\neural_upscaling\export_quicksrnet_custom.py --local-onnx-only
```

On Windows, these environment variables make the local trace noninteractive and
avoid console encoding issues from PyTorch's ONNX exporter:

```powershell
$env:QAIHM_CI = "1"
$env:QAIHM_STORE_ROOT = (Resolve-Path ".").Path
$env:PYTHONUTF8 = "1"
```

## Official Qualcomm TFLite Export

After configuring Qualcomm AI Hub Workbench:

```powershell
.\.venv-qai\Scripts\python.exe tools\neural_upscaling\export_quicksrnet_custom.py --runtime tflite --height 180 --width 320
```

Useful variants:

```powershell
# 320x180 -> 1280x720
.\.venv-qai\Scripts\python.exe tools\neural_upscaling\export_quicksrnet_custom.py --runtime tflite --height 180 --width 320

# 480x270 -> 1920x1080
.\.venv-qai\Scripts\python.exe tools\neural_upscaling\export_quicksrnet_custom.py --runtime tflite --height 270 --width 480

# QNN DLC for future Snapdragon-native experiments
.\.venv-qai\Scripts\python.exe tools\neural_upscaling\export_quicksrnet_custom.py --runtime qnn_dlc --height 180 --width 320
```
