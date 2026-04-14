# Majsoul YOLO Advisor

即時麻將輔助工具，透過螢幕截圖自動辨識雀魂手牌，計算向聴數、有效牌與打牌 EV，以懸浮視窗顯示建議。

---

## 解決的問題

雀魂等線上麻將平台無法直接取得牌局資料，玩家需要手動記憶手牌與場況。本專案透過：

1. **YOLOv8 即時辨識螢幕截圖中的手牌**，自動識別 34 種牌型（含寶牌紅五）
2. **自動計算向聴數與有效牌**，取代手動計算
3. **EV 排名推薦最佳棄牌**，考慮得點期望值與有效牌數
4. **懸浮視窗常駐顯示**，不干擾遊戲操作

---

## 核心 Features

### 辨識
- **YOLOv8n 手牌模型**：專為 13+1 手牌區域訓練，mAP50 ≈ 0.995
- **34 類牌型**：含寶牌紅五（5m/5p/5s 各有一般與紅牌兩種外觀）
- **多幀平滑**（Frame Smoother）：跨幀確認，降低誤偵測率
- **合成資料增強**：自動生成 1000+ 張合成手牌訓練圖，含寶牌機率加權（25%）
- 支援 3P / 4P 模式自動切換

### 麻將計算引擎
- **向聴數**（Shanten）：支援通常手、七對子、國士無雙，Numba JIT 加速
- **有效牌**（Effective Tiles）：列出所有能縮減向聴的牌及剩餘數量
- **EV 排名**（Simple EV）：`EV = 得點估算 × 有效牌數 / (向聴+1)`，即時顯示前三名棄牌建議
- **役種判定**：立直、平和、斷么、一盃口、役牌等常用役
- **符計算**：自動計算符數與點數估算
- **和牌判定**：自動偵測和牌，顯示役種、翻數與符數

### 輔助功能
- **預算 EV 快取**：持有 13 張牌等待摸牌時，背景預先計算所有可能摸牌的 EV，摸牌後即時顯示結果
- **手動修正**：點擊手牌牌格可修正辨識錯誤
- **寶牌管理**：手動新增 / 移除寶牌指示牌
- **副露記錄**：新增碰、吃、槓，納入 EV 計算
- **棄牌盤**：可拖曳標記已見棄牌，影響剩餘牌數計算

### 介面
- **永遠在最上層懸浮視窗**（Always-on-top，560×920）
- **F9 全域熱鍵**觸發分析，或開啟自動定時模式（可設秒數）
- **LiveView 即時預覽**：顯示辨識結果疊加的截圖
- **Skeleton 動畫**：計算中顯示骨架佔位，避免視覺跳動
- **UI 固定寬度**：動態資訊更新不造成版面位移

---

## Tech Stack

| 層級 | 技術 |
|------|------|
| 辨識模型 | YOLOv8n（Ultralytics）|
| 加速 | Numba JIT（Shanten DFS）、NumPy |
| 螢幕截圖 | mss（DXGI）/ win32api / PIL |
| GUI | Tkinter + ttk |
| 熱鍵 | Win32 `RegisterHotKey` |
| 訓練資料生成 | OpenCV 合成手牌圖、背景置換增強 |
| 語言 | Python 3.12 |
| GPU | CUDA（訓練用，推論 CPU / GPU 皆可）|

---

## 執行方式

### 環境需求

```
Python 3.12
CUDA（訓練用）/ CPU 推論即可
```

```bash
pip install ultralytics opencv-python numpy numba mss pillow
```

### 啟動輔助視窗（主要使用）

```bash
python windows_app.py
```

- 按 **F9** 截圖並分析，或勾選 **Auto** 開啟定時分析
- 若需指定手牌區域，點擊 ⚙ 設定選取區域

![demo_01](image/dezwv-l6801.gif)

![demo_02](image/4e96n-mmurf.gif)

---

### 訓練手牌模型（選用）

```bash
# 1. 生成合成訓練資料
python hand_synth.py --n 1000

# 2. 執行完整訓練 Pipeline（自動補資料 + Phase1 + Phase2 + 弱類別強化）
python run_hand.py
```

訓練完成的模型存放於：
```
runs/detect/majsoul_hand_phase2/weights/best.pt
```

### 推論測試（選用）

```bash
# 單張圖片推論並視覺化
python infer_hand.py

# 分析混淆矩陣
python check_misclass.py runs/detect/majsoul_hand_phase2/weights/best.pt
```

---

## 專案結構

```
majsoul_yolo/
├── windows_app.py          # 主程式：懸浮視窗 GUI（View 層）
├── src/
│   ├── advisor_controller.py   # Controller：執行緒管理、EV 快取
│   ├── mahjong_engine.py       # 向聴 / 有效牌 / 役種 / 符（Numba 加速）
│   ├── ev_engine.py            # EV 計算（Analytical / Monte Carlo）
│   ├── game_state.py           # 手牌 / 副露 / 棄牌狀態
│   ├── frame_smoother.py       # 多幀平滑確認
│   ├── screen_capture.py       # 螢幕截圖（mss / win32 / PIL）
│   └── tile_codec.py           # 牌 ID ↔ 類別名稱轉換
├── hand_synth.py           # 合成手牌訓練圖生成
├── train_hand.py           # 兩階段 YOLOv8 訓練
├── run_hand.py             # 完整訓練 Pipeline（自動補資料 + 弱類別強化）
├── infer_hand.py           # 推論 + 視覺化
└── dataset_hand/
    └── tile_images/        # 34 類牌圖（含寶牌紅五變體）
```
