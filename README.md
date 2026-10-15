# 50 Startups: Multiple Linear Regression & Feature Selection Explorer

[![CRISP-DM Methodology](https://img.shields.io/badge/Methodology-CRISP--DM-4f46e5.svg)](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining)
[![GitHub Pages](https://img.shields.io/badge/Live-GitHub%20Pages-10b981.svg)](https://kelp11211.github.io/AIoT_DIC2_MultipleLR/)
[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> 國立大學博士班課程：**物聯網數據分析與應用 (AIoT Data Analytics & Applications)**  
> **專案二 (DIC-2)：多元線性迴歸特徵選擇與奧卡姆剃刀模型評估**

🌐 **互動式儀表板線上展示 (Live Interactive Dashboard)**：  
👉 [https://kelp11211.github.io/AIoT_DIC2_MultipleLR/](https://kelp11211.github.io/AIoT_DIC2_MultipleLR/)

---

## 專案核心成果摘要 (Executive Summary)

本專案依循 **CRISP-DM（跨行業數據挖掘標準流程）** 生命週期，針對 50 家新創公司的營運費用（R&D 研發、Administration 行政管理、Marketing 行銷）與地理位置進行多元線性迴歸建模與特徵選擇，驗證統計學習中的**奧卡姆剃刀原則 (Occam's Razor)** 與**偏差-變異權衡 (Bias-Variance Tradeoff)**：

| 評估維度 | 全變數模型 ($k=5$, Full Model) | 最佳精簡模型 ($k=2$, Sweet Spot) | 決策意涵 (Decision Impact) |
| :--- | :--- | :--- | :--- |
| **納入特徵** | R&D, Admin, Marketing, State (NY, FL) | **R&D Spend, Marketing Spend** | 剔除雜訊變數，精準聚焦 |
| **測試集 RMSE** | \$9,055.96 ~ \$9,137.99 | **\$8,198.80 (全局最低)** | $k=5$ 遭遇過擬合峭壁 (Overfitting Cliff) |
| **決定係數 $R^2$** | ~ 0.8987 | **~ 0.9474 (全局最高)** | 解釋力顯著提升 |
| **調整後 $R^2$** | ~ 0.7721 | **~ 0.9324** | 懲罰冗餘參數後優勢更突出 |
| **F 檢定** | - | **$F = 349.0, p < 0.0001$** | 模型極具統計顯著性 |

### 核心擬合方程式 (Fitted OLS Model):
$$\text{Profit} = 46,346.34 + 0.7886 \times (\text{R\&D Spend}) + 0.0326 \times (\text{Marketing Spend})$$

- **R&D 研發支出**：每投入 \$1 美元，預期帶來約 **+\$0.79 美元**利潤 ($p < 0.0001$)，為首要關鍵驅動因子。
- **Administration 行政支出**：統計上不顯著 ($p > 0.05$)，草創與高成長階段擴張行政開銷對獲利無直接貢獻。
- **State 地理位置**：紐約、加州與佛州三地虛擬變數皆未達顯著水準 ($p > 0.05$)，地理聚落並非利潤差異關鍵。

---

## 專案檔案結構 (Project Structure)

```text
├── index.html                      # 互動式儀表板前端主頁面 (HTML5 + Modern CSS + Chart.js)
├── model_data.js                   # 5 種特徵選擇演算法與 k=1..5 基準數據庫
├── 50_Startups.py                  # 基線模型訓練與 CRISP-DM 流程腳本
├── verify_sweet_spot_evaluation.py # OLS 統計推論、VIF 檢定與模型精簡驗證腳本
├── FINAL_PROJECT_CONCLUSIONS.md    # 完整的 Phase 5 評估與結案決策建議報告
├── 50_Startups.csv                 # 50 家新創公司資料集 (50 rows, 5 columns)
├── startup_profit_lr.pkl           # 已訓練完成之 Scikit-Learn Pipeline 模型檔案
├── rd_spend_vs_profit.png          # 散佈圖：研發支出 vs 淨利
├── actual_vs_predicted_profit.png  # 評估圖：實際獲利 vs 預測獲利
├── residual_analysis.png           # 診斷圖：殘差分佈分析 (Homoscedasticity)
└── README.md                       # 專案說明文件
```

---

## 快速開始與重現 (Quick Start)

### 1. 執行基線模型訓練
```bash
python 50_Startups.py
```

### 2. 執行 OLS 統計推論與 VIF 共線性驗證
```bash
python verify_sweet_spot_evaluation.py
```

### 3. 本地預覽互動式 Dashboard
直接用瀏覽器開啟 `index.html`，或啟動 Python 本地伺服器：
```bash
python -m http.server 8080
# 瀏覽器開啟 http://localhost:8080
```

---

## 參考資料與致謝 (Acknowledgments)
- **Methodology**: CRISP-DM (Cross-Industry Standard Process for Data Mining)
- **Dataset**: [50 Startups Dataset (Avik Jain 100-Days-Of-ML)](https://github.com/Avik-Jain/100-Days-Of-ML-Code)
- **Design & Layout Reference**: [StatsUnlock Multiple Linear Regression Calculator](https://statsunlock.com/multiple-linear-regression-calculator/)
