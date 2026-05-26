# 推理引擎比較

## 比較矩陣

| 特性 | ORT GPU | ORT TRT EP | TensorRT FP32 | TensorRT FP16 |
|------|---------|------------|---------------|---------------|
| 格式 | .onnx | .onnx | .engine | .engine |
| 精度 | FP32 | FP32 / FP16 | FP32 | FP16 |
| 建置時間 | 無 | 首次執行時建置（快取後可略） | 數分鐘 | 數分鐘 |
| 引擎可攜性 | 跨平台 | 綁定 GPU 架構 | 綁定 GPU 架構 | 綁定 GPU 架構 |
| 記憶體用量 | 中 | 中（ORT overhead + TRT） | 中 | 低（約 50%）|
| 推理速度 | 基準 | 接近 TRT 原生 | 快 | 最快 |
| 精度損失 | 無 | 無（FP32 模式） | 無 | 極小 |
| API 複雜度 | 低 | 低（ORT 介面） | 中（trtexec CLI） | 中 |

## ORT TensorRT Execution Provider

ORT TRT EP 讓你**透過標準 ORT 介面**呼叫 TensorRT，省去直接操作 TRT Python API 的複雜度。

```python
sess = ort.InferenceSession(
    "model.onnx",
    providers=[
        ("TensorrtExecutionProvider", {
            "device_id": 0,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "./engines",
        }),
        "CUDAExecutionProvider",
    ]
)
```

### 與原生 TRT 的差異

```mermaid
graph LR
    subgraph "ORT TRT EP"
        A["應用程式"] --> B["ORT API"]
        B --> C["TRT EP 轉接層"]
        C --> D["TensorRT Runtime"]
    end
    subgraph "原生 TRT"
        E["應用程式"] --> F["TRT Python API<br/>（tensorrt + cuda-python）"]
        F --> G["TensorRT Runtime"]
    end
```

ORT TRT EP 多了一層轉接，通常會帶來 **0.5–2 ms** 額外開銷（視模型大小而定）；換來的是更簡潔的 API 且無需安裝 `tensorrt` Python bindings。

## 決策流程

```mermaid
flowchart TD
    Start(["需要推理"]) --> Q1{"需要跨 GPU<br/>可攜性？"}
    Q1 -->|是| ORT["ONNXRuntime GPU"]
    Q1 -->|否| Q2{"想用 TRT 加速<br/>但避免 TRT Python API？"}
    Q2 -->|是| ORTTRT["ORT TensorRT EP"]
    Q2 -->|否| Q3{"允許精度<br/>輕微損失？"}
    Q3 -->|否| TF32["TensorRT FP32"]
    Q3 -->|是| Q4{"記憶體受限<br/>或追求最高速？"}
    Q4 -->|是| TF16["TensorRT FP16"]
    Q4 -->|否| TF32
```

## 引擎相容性注意事項

TensorRT 引擎（包含 ORT TRT EP 快取的 `.engine`）**綁定**以下環境，跨環境需重新建置：
- GPU 架構（如 Ampere vs Ada Lovelace）
- TensorRT 版本
- CUDA 版本
