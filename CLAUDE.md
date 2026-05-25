# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking YOLO classification models (ONNX → TensorRT) on Windows with CUDA 12.8. The goal is to compare inference latency and throughput across ONNXRuntime GPU, TensorRT FP32, and TensorRT FP16.

## Environment

| Resource | Path |
|---|---|
| trtexec | `C:/Users/edisonhsieh/Downloads/TensorRT-10.8.0.43.Windows.win10.cuda-12.8/TensorRT-10.8.0.43/bin/trtexec.exe` |
| ONNX model | `C:/GPM_AI/H.onnx` |
| Test dataset | `C:/Users/edisonhsieh/Downloads/Test_dataset` |

## Setup

```powershell
# Create and activate uv environment
uv sync --group dev
uv run jupyter lab

# Install TensorRT Python bindings (manual step — not on PyPI for Windows)
uv pip install "C:/Users/edisonhsieh/Downloads/TensorRT-10.8.0.43.Windows.win10.cuda-12.8/TensorRT-10.8.0.43/python/tensorrt-10.8.0.43-cp312-none-win_amd64.whl"
```

## Key Files

- `benchmark.ipynb` — main benchmark notebook (10 cells, run top to bottom)
- `AI_Sample.zip` — contains `2Onnx.py` (YOLO → ONNX export) and `onnx_inference.py` (ORT classification inference reference)
- `engines/` — generated TRT engine files (created on first run)

## Model Details

- **Task**: image classification (9 classes: "1"–"9")
- **Preprocessing**: resize to model input size → divide by 255 → NCHW (no letterboxing)
- **Output**: logits or probabilities; apply softmax if `min < 0` or `sum ≠ 1`
- Input shape is auto-detected from the ONNX graph in Cell 3 (`MODEL_H`, `MODEL_W`, `MODEL_C`)

## Benchmark Notebook Sections

| Cell | Purpose |
|------|---------|
| 1 | Path constants (`TRTEXEC`, `ONNX_MODEL`, `TEST_DATASET`, `WARMUP_MS`, `DURATION_S`) |
| 2 | Verify GPU, CUDA, ORT providers, and all paths |
| 3 | ONNX model inspection (shape, opset, node count) |
| 4 | `trtexec` FP32 engine build + benchmark |
| 5 | `trtexec` FP16 engine build + benchmark |
| 6 | Parse latency/throughput from trtexec stdout |
| 7 | ONNXRuntime GPU latency baseline (300 runs) |
| 8 | TensorRT Python API engine inspection (requires TRT Python package) |
| 9 | Accuracy comparison ORT vs TRT FP16 on test images (requires TRT Python + `cuda-python`) |
| 10 | Summary table + bar charts → `benchmark_results.png` |

## Notes

- Cell 8 and Cell 9 require `tensorrt` Python bindings and `cuda-python`; they gracefully skip if not installed
- Engines are saved to `engines/H_fp32.engine` and `engines/H_fp16.engine`; delete them to force a rebuild
- The existing `onnx_inference.py` (in `AI_Sample.zip`) is the reference inference pipeline used to align preprocessing in Cell 9
