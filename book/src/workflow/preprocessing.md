# 前處理管線

## 管線步驟

```mermaid
flowchart LR
    A[原始影像<br/>任意尺寸] -->|PIL.Image.open| B[RGB 影像]
    B -->|resize| C[MODEL_H × MODEL_W]
    C -->|np.array / 255.0| D[FP32<br/>值域 0~1]
    D -->|transpose 2,0,1| E[CHW 格式]
    E -->|np.expand_dims| F[NCHW<br/>batch=1]
    F -->|astype float32| G[推理輸入]
```

## 關鍵特性

| 特性 | 設定 | 說明 |
|------|------|------|
| 縮放方式 | 直接 resize | **無 letterbox**，可能有形變 |
| 正規化 | ÷ 255 | 像素值映射到 [0, 1] |
| 通道順序 | RGB | 與 PyTorch 訓練一致 |
| 格式 | NCHW | TensorRT 標準輸入格式 |

## 輸出後處理

```mermaid
flowchart TD
    A[模型輸出 logits] --> B{min < 0 或<br/>sum ≠ 1？}
    B -->|是| C[套用 softmax]
    B -->|否| D[直接使用]
    C & D --> E[argmax → 類別索引]
    E --> F[類別標籤]
```

## 多後端前處理對齊

準確率比對的前提是各後端使用**完全相同**的前處理邏輯：resize 方式、正規化係數、通道順序必須與訓練時保持一致。任何差異都會導致輸出不一致，使準確率下降。
