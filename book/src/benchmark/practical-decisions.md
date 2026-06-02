# 實務決策概念

本頁整理 YOLO + TensorRT 實際部署時的決策思路，幫助在類似專案中做出正確取捨。

## 決策流程圖

```mermaid
flowchart TD
    A[開始優化] --> B[先量測 FP32 baseline]
    B --> C[加 --fp16 測試]
    C --> D{FP16 夠快了嗎？}
    D -->|是| E[停在 FP16，進入部署]
    D -->|否| F[評估 INT8 成本]
    F --> G{"能接受 Calibration<br/>+ mAP 驗證成本？"}
    G -->|是| H[INT8 + Calibrator]
    G -->|否| E

    E --> I{場景是什麼？}
    I -->|低延遲| J["batch=1<br/>opt 對齊 batch=1"]
    I -->|高吞吐量| K["batch=8~32<br/>opt 對齊實際分佈"]
```

## 決策原則 1：先量測，再優化

很多人上來就問「要不要 INT8」，但正確第一步是量測：

```bash
# FP32 baseline
trtexec --onnx=yolo.onnx --optShapes=images:1x3x640x640

# FP16（幾乎免費）
trtexec --onnx=yolo.onnx --fp16 --optShapes=images:1x3x640x640
```

如果 FP16 已經夠快，就不需要 INT8 的複雜度。**過早優化是最常見的浪費。**

## 決策原則 2：低延遲 vs 高吞吐量是不同的 Engine

這是最重要的概念分歧，同一個 ONNX 應該 build 兩個不同 Engine：

```mermaid
graph LR
    subgraph "低延遲場景"
        A1["即時攝影機<br/>單一請求"]
        A2["batch=1<br/>opt 對齊 batch=1<br/>目標：單張 < 10ms"]
        A1 --> A2
    end

    subgraph "高吞吐量場景"
        B1["離線影片處理<br/>API 服務"]
        B2["batch=8~32<br/>排隊攢夠再推<br/>目標：最大化 GPU 使用率"]
        B1 --> B2
    end
```

OptimizationProfile 的 `opt` 對齊你的實際使用情境，TensorRT 才能針對那個形狀做最佳 kernel 選擇。

## 決策原則 3：FP16 幾乎是免費的

```
FP32（baseline）
  ↓ 加 --fp16，幾乎零風險
FP16（推薦預設）  ← 大部分專案停在這就夠了
  ↓ 如果還需要更快，且能接受驗證成本
INT8（需要 Calibration dataset + mAP 驗證）
```

YOLO 系列用 FP16 通常 mAP 差距 < 0.5%，速度快 1.5–2 倍，沒有理由不開。

## 決策原則 4：Engine 綁定 GPU，這個常常被忽略

`.engine` 不能跨 GPU 架構使用：

```
開發機（RTX 3090）build 的 engine ≠ 上線機（T4）能用的 engine

正確做法：
  方案 A：CI/CD pipeline 在目標 GPU 上 build engine
  方案 B：部署時帶著 ONNX，第一次啟動時 build & cache
```

## 決策原則 5：OptimizationProfile 的 opt 要對齊真實分佈

```python
profile.set_shape("images",
    min=(1,  3, 640, 640),
    opt=(4,  3, 640, 640),   # ← 這個影響最大
    max=(16, 3, 640, 640),
)
```

如果 90% 的請求是 batch=4，`opt` 就填 4，不要填 1 或填 max。填錯了整個 profile 的效能都會偏掉。

## 決策原則 6：INT8 Calibration 資料要有代表性

如果最終需要 INT8：

```python
class YOLOCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, data_loader, cache_file):
        self.data_loader = iter(data_loader)
        self.cache_file  = cache_file
        self.d_input     = cuda.mem_alloc(batch_bytes)

    def get_batch(self, names):
        try:
            batch = next(self.data_loader)
            cuda.memcpy_htod(self.d_input, batch)
            return [int(self.d_input)]
        except StopIteration:
            return None
```

Calibration 資料必須涵蓋你的實際場景（白天/夜晚、近景/遠景）。

> **常見錯誤**：用訓練集做 calibration。應使用接近上線分佈的資料。

## 場景決策矩陣

| 場景 | batch | 精度 | 動態 Shape | 備注 |
|------|-------|------|-----------|------|
| 即時攝影機（1 路） | 1 | FP16 | 固定即可 | 延遲優先 |
| 多路 NVR（8–16 路） | 8–16 | FP16 | 動態 batch | 吞吐優先 |
| 離線影片分析 | 32+ | INT8 | 固定 | 精度驗證後再上 |
| 醫療/工業缺陷檢測 | 1 | FP16 或 FP32 | 固定 | mAP 容忍度低 |
| API 服務（變動 QPS） | 動態 | FP16 | 動態 batch | 搭配 Triton |

## Benchmark 腳本化比較

要比較不同參數效能，建議腳本化執行，省去重複 build 的時間：

```bash
# 只 build 一次，存 engine
trtexec --onnx=yolo.onnx --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:4x3x640x640 \
  --maxShapes=images:16x3x640x640 \
  --saveEngine=yolo_fp16.engine

# 之後測不同 batch，不用重新 build
trtexec --loadEngine=yolo_fp16.engine --shapes=images:1x3x640x640
trtexec --loadEngine=yolo_fp16.engine --shapes=images:4x3x640x640
trtexec --loadEngine=yolo_fp16.engine --shapes=images:8x3x640x640
```

FP32 vs FP16 需要各 build 一次，但不同 batch size 不用重複 build。詳見 [評測方法論](methodology.md)。

## 決策原則 7：CUDA Graph 可消除 CPU Launch 開銷

固定 batch size（非動態）且追求極致延遲時，CUDA Graph 是值得的：

```mermaid
flowchart TD
    A{"輸入形狀固定嗎？"}
    A -->|是| B{"延遲 < 1ms 仍不夠快？"}
    A -->|否| Z["不適用 CUDA Graph<br/>動態 shape 無法捕捉"]
    B -->|是| C["啟用 CUDA Graph<br/>context.capture_begin / end"]
    B -->|否| D["不需要，現有延遲已足夠"]

    C --> E["每次 infer 改用<br/>graph.launch 取代 execute_async"]
```

CPU dispatch 開銷在 batch=1 的小模型可佔總延遲 20–40%。啟用後通常節省 0.1–0.5ms。

```python
# 捕捉 graph（只做一次）
with torch.cuda.graph(graph):
    context.execute_async_v3(stream_handle=stream.handle)

# 推理時直接 replay
graph.replay()
```

> **注意**：捕捉期間輸入 buffer 必須與推理期間相同。

## 決策原則 8：Workspace 大小影響 Kernel 選擇

`--workspace` 控制 TensorRT 在 Tactic 選擇時可使用的臨時 GPU 記憶體：

```
太小（< 256 MB）→ 部分高效 kernel 無法被選中 → 性能下降
太大（> 4 GB）  → 擠壓其他模型或 OOM → 系統不穩
推薦：512 MB–2 GB，視 GPU 總 VRAM 決定
```

```bash
# 指定 workspace（單位 MiB，不是 MB）
trtexec --onnx=yolo.onnx --fp16 --workspace=1024
```

| GPU VRAM | 建議 workspace |
|----------|--------------|
| 4 GB | 256–512 MB |
| 8 GB | 512 MB–1 GB |
| 16 GB+ | 1–2 GB |

## 決策原則 9：多 Stream 提升 GPU 使用率（吞吐量場景）

低延遲場景用 1 stream，高吞吐量場景可跑多個 stream 同時 infer：

```mermaid
graph LR
    subgraph "單 Stream（延遲優先）"
        S1["Stream 0"] --> I1["Infer"] --> S1
    end

    subgraph "多 Stream（吞吐優先）"
        M1["Stream 0"] --> J1["Infer A"]
        M2["Stream 1"] --> J2["Infer B"]
        M3["Stream 2"] --> J3["Infer C"]
        J1 & J2 & J3 --> R["結果合併"]
    end
```

每個 stream 需要獨立的 `IExecutionContext` 與 I/O buffer。context 是執行緒不安全的，不能共用。

## 效能不如預期時的診斷流程

```mermaid
flowchart TD
    P["延遲比預期高"] --> Q{"與 trtexec 基準比，差多少？"}
    Q -->|"> 50%"| R["CPU Overhead 或 GPU 搶佔"]
    Q -->|"< 20%"| S["在可接受範圍，tuning 邊際效益低"]
    Q -->|"20–50%"| T["值得調查"]

    R --> R1{"用了 CUDA Graph？"}
    R1 -->|否| R2["試開 CUDA Graph<br/>消除 CPU launch"]
    R1 -->|是| R3["檢查是否有其他程序<br/>搶佔 GPU（nvidia-smi）"]

    T --> T1{"Profile 哪層最慢？"}
    T1 --> T2["trtexec --profilingVerbosity=detailed<br/>找瓶頸層"]
    T2 --> T3{"是 plugin 或 unsupported op？"}
    T3 -->|是| T4["考慮 TRT Plugin 或<br/>改寫成可 fuse 的寫法"]
    T3 -->|否| T5["換 workspace 大小<br/>或換 tacticSources"]
```
