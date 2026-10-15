# AIoT Multi-Problem Linear Regression & Feature Selection Explorer

[![CRISP-DM Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-4f46e5.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![GitHub Pages](https://img.shields.io/badge/Live-GitHub%20Pages-10b981.svg)](https://kelp11211.github.io/AIoT_DIC2_MultipleLR/)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 國立大學博士班課程：**物聯網數據分析與應用 (AIoT Data Analytics & Applications)**  
> **專案二 (DIC-2)：多元線性迴歸特徵選擇、奧卡姆剃刀原則與多問題互動式儀表板**

🌐 **互動式多問題切換儀表板線上展示 (Live Multi-Problem Dashboard)**：  
👉 **[https://kelp11211.github.io/AIoT_DIC2_MultipleLR/](https://kelp11211.github.io/AIoT_DIC2_MultipleLR/)**

---

## 雙問題架構與核心成果摘要 (Two Machine Learning Problems)

本專案依循 **CRISP-DM（跨行業數據挖掘標準流程）** 生命週期，建立統一的特徵選擇（Lasso, Forward SFS, RFE, ElasticNet, Exhaustive Search）與奧卡姆剃刀模型評估體系，支援頂部一鍵無縫切換兩大經典機器學習迴歸問題：

### 問題一：50 Startups 新創企業獲利預測
針對 50 家新創公司的營運費用（R&D 研發、Administration 行政管理、Marketing 行銷）與地理位置進行建模：
- **全局最佳甜蜜點 (Sweet Spot, $k=2$)**：`R&D Spend` + `Marketing Spend`
- **測試集 RMSE**：**\$8,198.80 (全局最低)**，相較於 $k=5$ 全特徵模型 (\$9,055 ~ \$9,138) 避免了過擬合峭壁。
- **決定係數 $R^2$**：**0.9474** (解釋 94.74% 利潤變異)
- **核心擬合方程式**：
  $$	ext{Profit} = 46,346.34 + 0.7886 	imes (	ext{R\&D Spend}) + 0.0326 	imes (	ext{Marketing Spend})$$

### 問題二：Boston Housing 波士頓房價估算
針對波士頓 506 個街區之空間結構 (RM)、環境學區 (PTRATIO) 與社區經濟地位 (LSTAT) 進行房屋中位數定價建模：
- **全局最佳甜蜜點 (Sweet Spot, $k=3$)**：`RM` (房間數) + `PTRATIO` (生師比) + `LSTAT` (中低所得人口比例)
- **測試集 RMSE**：**\$5.21k / \$5,207 (全局最低)**，有效剔除 TAX 與 RAD 的嚴重多重共線性 (VIF > 7.5)。
- **決定係數 $R^2$**：**0.6303** (測試集解釋率)
- **核心擬合方程式**：
  $$	ext{MEDV} = 14.5881 + 4.9331 	imes (	ext{RM}) - 0.8671 	imes (	ext{PTRATIO}) - 0.5671 	imes (	ext{LSTAT})$$

---

## 專案檔案結構 (Project Structure)

```text
├── index.html                      # 支援雙問題切換之未來科技感互動式儀表板 (HTML5 + Dark Glassmorphism)
├── model_data.js                   # 包含 50 Startups 與 Boston Housing 之多演算法基準資料庫
├── 50_Startups.py                  # 問題一：50 Startups 基線模型訓練腳本
├── 50_Startups.csv                 # 問題一資料集
├── verify_sweet_spot_evaluation.py # 問題一 OLS 統計推論、VIF 檢定與特徵精簡驗證程式
├── FINAL_PROJECT_CONCLUSIONS.md    # 問題一 CRISP-DM 評估與決策報告
├── boston_housing.py               # 問題二：Boston Housing CRISP-DM 完整流程與訓練腳本
├── BostonHousing.csv               # 問題二資料集 (506 筆街區資料)
├── BOSTON_HOUSING_CONCLUSIONS.md   # 問題二 CRISP-DM 房產估價與都市規劃決策報告
├── startup_profit_lr.pkl           # 問題一已訓練儲存之 Pipeline 模型
├── boston_housing_lr.pkl           # 問題二已訓練儲存之 Pipeline 模型
├── *.png                           # 高解析評估圖表 (實際 vs 預測、殘差分析、特徵散佈圖)
└── README.md                       # 專案總說明文件
```

---

## 快速執行與驗證 (Quick Start)

### 1. 執行 50 Startups 流程
```bash
python 50_Startups.py
python verify_sweet_spot_evaluation.py
```

### 2. 執行 Boston Housing 流程
```bash
python boston_housing.py
```

### 3. 本地預覽互動式 Dashboard
直接用瀏覽器開啟 `index.html`，或啟動 Python 本地伺服器：
```bash
python -m http.server 8088
# 瀏覽器開啟 http://localhost:8088
```

---

## 參考資料與致謝 (Acknowledgments)
- **Methodology**: CRISP-DM (Cross-Industry Standard Process for Data Mining)
- **Datasets**: 50 Startups Dataset & Boston Housing Dataset
