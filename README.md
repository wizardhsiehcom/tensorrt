# TensorRT Benchmark — YOLO Classification on Windows

Benchmarking a YOLO image-classification model (ONNX → TensorRT) on **Windows + CUDA 12.8**, comparing inference latency and throughput across ONNXRuntime and TensorRT at multiple precisions.

## Key Results

Hardware: **NVIDIA RTX 5070 Laptop (Blackwell, SM 12.0)**

| Backend | Mean Latency | vs ORT CPU |
|---------|-------------|------------|
| ORT CPU (baseline) | 36.4 ms | 1× |
| TRT FP32 | 3.3 ms | ~11× |
| TRT FP16 | 1.3 ms | ~28× |
| TRT INT8 | 0.9 ms | ~40× |

## Setup

```powershell
uv sync --group dev
cp .env.example .env   # update TRTEXEC, ONNX_MODEL, TEST_DATASET
uv run jupyter lab
```

Run notebooks `01` → `08` in order. Engines are cached under `engines/`.

## Knowledge Graph

```powershell
mdbook serve book --open
```
