# Engine 參數 Sweep 調研

大規模參數掃描（Parameter Sweep）是找出最佳 TensorRT engine 配置的系統化方法。
本頁說明掃描哪些維度、如何解讀 Sweep 輸出，以及如何轉換成部署決策。

## 掃描維度（四個維度）

| 維度 | 選項 | GPU 要求 | 說明 |
|------|------|---------|------|
| `precision` | fp32 | 所有 GPU | 完整精度 |
| | fp16 | Pascal+ | 最常用的加速方案 |
| | bf16 | Hopper+ / Blackwell | Brain Float 16 |
| | int8 | Turing+ | 8-bit 整數，需注意精度 |
| | fp8  | Ada+ / Blackwell | 8-bit 浮點（E4M3） |
| `builderOptimizationLevel` | 0, 2, 4, 5 | — | 0 = 快速 build；5 = 最激進優化（build 最慢） |
| `workspace_mb` | 256, 1024 | — | GPU workspace 上限；影響 layer fusion 搜尋空間 |
| `batch_mode` | static | — | 固定 batch=1，與原始 ONNX 一致 |
| | dynamic | 動態 ONNX | min=1 / opt=4 / max=8；需 ONNX batch dim 為動態 |

四個維度交叉產生 5 × 4 × 2 × 2 = **80 組**配置，  
每組約 2–5 分鐘，全跑完約 2.7–6.7 小時。

> **縮減建議**：如需快速驗證，可先只保留 `precision=["fp16","int8"]`、  
> `builder_opt_level=[0,4]`、`batch_mode=["static"]`，共 4 組，約 10–20 分鐘。

---

## 動態 Batch 說明

`batch_mode=dynamic` 時，trtexec 加入形狀限制旗標讓 engine 接受可變 batch：

```
--minShapes=images:1x3x448x448
--optShapes=images:4x3x448x448
--maxShapes=images:8x3x448x448
--shapes=images:4x3x448x448      # benchmark 以 opt batch 量測
```

> **前置條件**：ONNX 模型的 batch dim 必須是動態（`dim_param`）。  
> 若模型為靜態 batch=1（`dim_value=1`），dynamic 組別在 build 時會失敗並自動跳過。  
> 詳見 [動態 Batch 工作流程](../workflow/dynamic-batch.md)。

---

## 加速技巧：Timing Cache

```mermaid
flowchart LR
    C1["Config 0<br/>fp32 static"] -->|"寫入 timing.cache"| TC[("timing.cache")]
    C2["Config 1<br/>fp16 static"] -->|"讀取已知 kernel timing"| TC
    C3["Config N<br/>..."] -->|"命中 cache → 跳過重測"| TC
    TC -->|"後期 build 顯著加快"| DONE["全部完成"]
```

`--timingCacheFile` 跨所有 build 共用，後期 config 的 build 時間可縮短 50–80%。

---

## 結果解讀

### Pivot Table（依 workspace × batch_mode 分群）

```
=== workspace=1024MB  batch=static — Mean Latency (ms) ===
precision          bf16   fp16   fp32   fp8   int8
builder_opt_level
0                  3.xx   2.07   4.11   3.xx   1.xx
2                  3.xx   1.52   3.60   3.xx   1.xx
4                  3.xx   1.40   3.48   3.xx   0.xx
5                  3.xx   1.43   3.44   3.xx   0.xx
```

**橫向**（同 row）比 precision：找最快的精度類型  
**縱向**（同 column）比 opt_level 的回報遞減：opt=4→5 通常改善 < 0.1ms，但 build 時間可多 2–3 倍  
**兩張表對比** workspace 256 vs 1024：差距小 → workspace 不是瓶頸，選小的省 GPU 記憶體

---

### Heatmap（視覺化版 Pivot）

```mermaid
graph LR
    subgraph "顏色對應"
        L["淺黃色<br/>latency 低（好）"]
        D["深紅色<br/>latency 高（差）"]
    end
```

一眼找出最佳格子（最淺色），通常落在 **INT8 或 FP16 × 高 opt_level** 的交叉點。  
BF16 和 FP8 在 Blackwell 上往往呈現深色（詳見 [精度全覽比較](precision-sweep.md)）。

---

### Pareto Scatter

這是最重要的決策圖。

```mermaid
graph TD
    subgraph "Pareto Scatter 解讀"
        A["X 軸 = Build Time（部署成本）"]
        B["Y 軸 = Mean Latency（推論速度）"]
        C["理想點 = 左下角<br/>（build 快 且 推論快）"]
        A --- C
        B --- C
    end
```

**Pareto frontier** 是「無法在不犧牲一方的情況下再改善另一方」的那條邊界。

| 觀察 | 決策 |
|------|------|
| opt=5 比 opt=4 只快 0.05ms，但多花 120s build | opt=4 是更好的取捨 |
| fp16 opt=2 在 frontier 上 | 這是「build 效率最佳」的點 |
| workspace 1024 與 256 幾乎重疊 | workspace 不影響此模型，選 256 |

---

### Throughput Bar

QPS（Queries Per Second）衡量批次吞吐。

| 使用場景 | 主要指標 |
|---------|---------|
| 即時推論（單張、低延遲） | `mean_ms`（越低越好） |
| 批次處理、API server | `throughput_qps`（越高越好） |

---

## `mean_ms` vs `gpu_compute_ms` 的差距

```mermaid
flowchart LR
    CPU["CPU 端"] -->|"H2D memcpy"| GPU["GPU 計算<br/>gpu_compute_ms"]
    GPU -->|"D2H memcpy"| CPU2["CPU 端結果"]
    CPU -->|"端對端"| E2E["mean_ms"]
```

- `mean_ms - gpu_compute_ms` 就是 **I/O 搬移開銷**
- 差距大（> 0.3ms）→ I/O 是瓶頸，考慮 CUDA Graph / Pinned Memory
- 差距小 → 瓶頸在計算本身，繼續調 precision 或 opt_level

---

## 決策框架

```mermaid
flowchart TD
    S["拿到 Sweep 結果"] --> P{"看 Pareto 圖"}
    P --> F["找 Pareto frontier 上的點"]
    F --> Q{"部署環境能接受多長的 build time？"}
    Q -->|"不限（固定部署）"| OPT5["選 opt_level=5<br/>最低 latency"]
    Q -->|"< 60s（CI/CD 重 build）"| OPT2["選 Pareto frontier 上<br/>build 最快的點"]
    OPT5 --> ACC{"需要精度驗證？"}
    OPT2 --> ACC
    ACC -->|"FP16 / INT8 精度 OK"| DONE["鎖定最佳 config"]
    ACC -->|"精度不可接受"| FP32["改用 FP32 最佳 opt_level"]
    DONE --> WS{"workspace 影響大嗎？"}
    FP32 --> WS
    WS -->|"兩張 heatmap 差不多"| W256["選 256MB，省 GPU 記憶體"]
    WS -->|"1024MB 明顯更快"| W1024["選 1024MB"]
```

## 情境決策矩陣

| 情境 | 推薦 precision | 推薦 opt_level | 推薦 workspace |
|------|--------------|--------------|--------------|
| 固定部署，不重 build | INT8（驗證後）或 fp16 | 5 | 256MB（除非明顯更快） |
| CI/CD 頻繁重 build | fp16 | 2 | 256MB |
| 精度要求嚴格（AOI / 醫療） | fp32 或 fp16（驗證後） | 4 | 256MB |
| GPU 記憶體緊張（< 4GB） | fp16 | 2 | 256MB |
| 最大化批次吞吐 | INT8（驗證後） | 5 | 1024MB |
| 動態 batch 推論 | fp16 | 4 | 1024MB |

---

## 與其他調研頁面的關係

```mermaid
flowchart LR
    PS["[精度全覽]<br/>5 種精度<br/>固定預設參數"]
    SW["[參數 Sweep]<br/>80 組配置<br/>4 維度交叉"]
    BS["[Batch Sweep]<br/>batch 1–32<br/>動態 engine"]
    FC["[最終比較]<br/>各精度 Sweep 最佳<br/>vs ORT 基線"]

    PS -->|"確認精度可行性"| SW
    SW -->|"輸出最佳 config"| FC
    BS -->|"確認 batch 效益"| FC
```

> Sweep 輸出存於 `sweep/sweep_results.csv`，視覺化圖表見 `sweep/sweep_results.png`。
