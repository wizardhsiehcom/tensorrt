# Engine 參數 Sweep 調研

大規模參數掃描（Parameter Sweep）是找出最佳 TensorRT engine 配置的系統化方法。
本頁說明掃描哪些維度、如何解讀 `05_param_sweep.ipynb` 的輸出，以及如何轉換成部署決策。

## 掃描維度

| 參數 | 典型選項 | 說明 |
|------|---------|------|
| `precision` | fp32, fp16 | 計算精度；fp16 幾乎免費加速 |
| `builderOptimizationLevel` | 0, 2, 4, 5 | 0 = 快速 build；5 = 最激進優化（build 最慢） |
| `workspace_mb` | 256, 1024 | GPU workspace 上限；影響 layer fusion 搜尋空間 |

三個維度交叉產生 2 × 4 × 2 = **16 組**配置，每組約 2–5 分鐘，全跑完約 30–80 分鐘。

## 加速技巧：Timing Cache

```mermaid
flowchart LR
    C1["Config 0<br/>fp32 opt=0"] -->|"寫入 timing.cache"| TC[("timing.cache")]
    C2["Config 1<br/>fp32 opt=2"] -->|"讀取已知 kernel timing"| TC
    C3["Config N<br/>..."] -->|"命中 cache → 跳過重測"| TC
    TC -->|"後期 build 顯著加快"| DONE["全部完成"]
```

`--timingCacheFile` 跨所有 build 共用，後期 config 的 build 時間可縮短 50–80%。

---

## 結果解讀

### Pivot Table（Cell 5）

```
=== workspace = 256 MB — Mean Latency (ms) ===
precision          fp16    fp32
builder_opt_level
0                  1.45    3.20
2                  1.38    3.05
4                  1.31    2.98
5                  1.29    2.97
```

**橫向**（同 row）比 precision：
- fp32 ÷ fp16 = 加速倍率，通常 1.5–2.5×

**縱向**（同 column）比 opt_level 的回報遞減：
- opt=0 → 2：改善明顯
- opt=4 → 5：改善通常 < 0.1ms，但 build 時間可能多 2–3 倍

**兩張表對比** workspace 256 vs 1024：
- 若數字幾乎相同，workspace 不是瓶頸，選小的省 GPU 記憶體

---

### Heatmap（Cell 6，圖一二）

```mermaid
graph LR
    subgraph "顏色對應"
        L["淺黃色<br/>latency 低（好）"]
        D["深紅色<br/>latency 高（差）"]
    end
```

一眼找出最佳格子（最淺色），通常落在 **fp16 × 高 opt_level** 的右下角。

---

### Pareto Scatter（Cell 6，圖三）

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
落在 frontier 上的點才值得認真考慮；frontier 右上方的點被其他配置完全支配。

| 觀察 | 決策 |
|------|------|
| opt=5 比 opt=4 只快 0.05ms，但多花 120s build | opt=4 是更好的取捨 |
| fp16 opt=2 在 frontier 上 | 這是「build 效率最佳」的點 |
| workspace 1024 與 256 幾乎重疊 | workspace 不影響此模型，選 256 |

---

### Throughput Bar（Cell 6，圖四）

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
    S[拿到 Sweep 結果] --> P{看 Pareto 圖}
    P --> F["找 Pareto frontier 上的點"]
    F --> Q{"部署環境能接受<br/>多長的 build time？"}
    Q -->|"不限（固定部署）"| OPT5["選 opt_level=5<br/>最低 latency"]
    Q -->|"< 60s（CI/CD 重 build）"| OPT2["選 Pareto frontier 上<br/>build 最快的點"]
    OPT5 --> ACC{"需要精度驗證？"}
    OPT2 --> ACC
    ACC -->|FP16 精度 OK| DONE["鎖定 FP16 最佳 config"]
    ACC -->|FP16 精度不可接受| FP32["改用 FP32 最佳 opt_level"]
    DONE --> WS{"workspace 影響大嗎？"}
    FP32 --> WS
    WS -->|"兩張 heatmap 差不多"| W256["選 256MB，省 GPU 記憶體"]
    WS -->|"1024MB 明顯更快"| W1024["選 1024MB"]
```

## 情境決策矩陣

| 情境 | 推薦 precision | 推薦 opt_level | 推薦 workspace |
|------|--------------|--------------|--------------|
| 固定部署，不重 build | fp16 | 5 | 256MB（除非明顯更快） |
| CI/CD 頻繁重 build | fp16 | 2 | 256MB |
| 精度要求嚴格（AOI/醫療） | fp32 或 fp16（驗證後） | 4 | 256MB |
| GPU 記憶體緊張（< 4GB） | fp16 | 2 | 256MB |
| 最大化批次吞吐 | fp16 | 5 | 1024MB |

> 完整 Sweep 輸出存於 `sweep/sweep_results.csv`，視覺化圖表見 `sweep/sweep_results.png`。
> 相關實作見 `05_param_sweep.ipynb`。
