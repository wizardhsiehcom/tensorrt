# 整體架構

## 系統元件關係圖

```mermaid
graph TD
    subgraph "模型來源"
        A[PyTorch YOLO 模型]
    end

    subgraph "轉換層"
        B[ONNX 格式]
    end

    subgraph "推理後端"
        C[ONNXRuntime<br/>CUDAExecutionProvider]
        D[TensorRT FP32]
        E[TensorRT FP16]
    end

    subgraph "評測輸出"
        F[延遲 ms]
        G[吞吐量 QPS]
        H[準確率]
    end

    A -->|ONNX 匯出| B
    B --> C
    B -->|trtexec| D
    B -->|trtexec --fp16| E
    C --> F & G & H
    D --> F & G & H
    E --> F & G & H
```

## 元件說明

### trtexec
NVIDIA 提供的命令列工具，負責：
1. 將 ONNX 解析為 TensorRT 網路
2. 最佳化並序列化為 `.engine` 檔案
3. 內建效能基準測試

### TensorRT Python API
用於反序列化引擎並執行推理：
- 需要 `tensorrt` wheel + `cuda-python`
- 支援手動控制記憶體與 stream 執行

### ONNXRuntime GPU
- 透過 `CUDAExecutionProvider` 在 GPU 執行
- 作為評測基準線（不需額外轉換）
