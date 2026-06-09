from pathlib import Path
import os, re

# D:\tensorrt\code\src\env.py  →  code/src  →  code  →  repo root
_REPO = Path(__file__).resolve().parent.parent.parent


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ[k.strip()] = v.strip()


_load_env(_REPO / ".env")

TRTEXEC      = Path(os.environ.get("TRTEXEC",      "C:/Users/edisonhsieh/Downloads/TensorRT-10.8.0.43.Windows.win10.cuda-12.8/TensorRT-10.8.0.43/bin/trtexec.exe"))
ONNX_MODEL   = Path(os.environ.get("ONNX_MODEL",   "C:/GPM_AI/H.onnx"))
TEST_DATASET = Path(os.environ.get("TEST_DATASET",  "D:/data/Test_dataset"))

ENGINES_DIR = Path("engines")
ENGINE_FP32 = ENGINES_DIR / "H_fp32.engine"
ENGINE_FP16 = ENGINES_DIR / "H_fp16.engine"

TRT_LIB = TRTEXEC.parent.parent / "lib"
if TRT_LIB.exists():
    _trt_lib_str = str(TRT_LIB)
    if _trt_lib_str not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _trt_lib_str + os.pathsep + os.environ.get("PATH", "")

WARMUP_MS  = 500
DURATION_S = 10
ORT_RUNS   = 300


def setup_matplotlib() -> None:
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    _font = _REPO / "book/src/fonts/Cubic_11.ttf"
    if _font.exists():
        fm.fontManager.addfont(str(_font))
        _prop = fm.FontProperties(fname=str(_font))
        _name = _prop.get_name()
        plt.rcParams["font.family"]        = [_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"Font loaded: {_name}  ({_font})")
    else:
        print(f"WARNING: Font not found: {_font}")
        print("中文字形可能無法正常顯示")


def parse_trtexec(stdout: str) -> dict:
    patterns = {
        "mean_ms":        r"mean\s*=\s*([\d.]+)\s*ms",
        "median_ms":      r"median\s*=\s*([\d.]+)\s*ms",
        "p95_ms":         r"95th percentile.*?=\s*([\d.]+)\s*ms",
        "p99_ms":         r"99th percentile.*?=\s*([\d.]+)\s*ms",
        "throughput_qps": r"Throughput:\s*([\d.]+)\s*qps",
        "gpu_compute_ms": r"GPU Compute Time.*?mean\s*=\s*([\d.]+)\s*ms",
    }
    return {
        k: float(m.group(1))
        for k, pat in patterns.items()
        if (m := re.search(pat, stdout, re.IGNORECASE))
    }


__all__ = [
    "TRTEXEC", "ONNX_MODEL", "TEST_DATASET",
    "ENGINES_DIR", "ENGINE_FP32", "ENGINE_FP16",
    "TRT_LIB",
    "WARMUP_MS", "DURATION_S", "ORT_RUNS",
    "setup_matplotlib", "parse_trtexec",
]
